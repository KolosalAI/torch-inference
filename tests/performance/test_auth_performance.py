"""
Performance tests for the authentication system.

This module tests the performance characteristics of the auth system.
"""

import pytest
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
from pathlib import Path
from statistics import mean, median
from unittest.mock import patch

from framework.auth import (
    JWTHandler, UserStore, AuthMiddleware, 
    hash_password, generate_api_key, generate_password
)
from framework.auth.models import User


class TestAuthPerformance:
    """Test authentication system performance."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def jwt_handler(self):
        """Create JWT handler for performance testing."""
        return JWTHandler(
            secret_key="performance_test_secret_key",
            algorithm="HS256",
            access_token_expire_minutes=30,
            refresh_token_expire_days=7
        )
    
    @pytest.fixture
    def user_store(self, temp_dir):
        """Create user store for performance testing."""
        users_file = temp_dir / "perf_users.json"
        api_keys_file = temp_dir / "perf_keys.json"
        return UserStore(str(users_file), str(api_keys_file))
    
    @pytest.fixture
    def auth_middleware(self, jwt_handler, user_store):
        """Create auth middleware for performance testing."""
        return AuthMiddleware(jwt_handler, user_store)
    
    def measure_time(self, func, *args, **kwargs):
        """Measure execution time of a function."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        return result, end_time - start_time
    
    def test_password_hashing_performance(self):
        """Test password hashing performance."""
        passwords = [
            "simple",
            "ComplexPassword123!",
            "VeryLongPasswordWithManyCharacters123!@#$%^&*()",
            "Unicode密码测试123!",
        ]
        
        times = []
        for password in passwords:
            _, duration = self.measure_time(hash_password, password)
            times.append(duration)
        
        # Password hashing should complete within reasonable time
        max_time = max(times)
        avg_time = mean(times)
        
        assert max_time < 1.0, f"Password hashing took too long: {max_time:.3f}s"
        assert avg_time < 0.5, f"Average password hashing time too high: {avg_time:.3f}s"
        
        print(f"Password hashing - Max: {max_time:.3f}s, Avg: {avg_time:.3f}s")
    
    def test_jwt_token_creation_performance(self, jwt_handler):
        """Test JWT token creation performance."""
        user_data = {
            "sub": "user_123",
            "username": "perfuser",
            "email": "perf@example.com",
            "roles": ["user"]
        }
        
        # Test access token creation
        access_times = []
        for _ in range(100):
            _, duration = self.measure_time(jwt_handler.create_access_token, user_data)
            access_times.append(duration)
        
        # Test refresh token creation
        refresh_times = []
        for _ in range(100):
            _, duration = self.measure_time(jwt_handler.create_refresh_token, user_data)
            refresh_times.append(duration)
        
        avg_access_time = mean(access_times)
        avg_refresh_time = mean(refresh_times)
        
        assert avg_access_time < 0.01, f"Access token creation too slow: {avg_access_time:.4f}s"
        assert avg_refresh_time < 0.01, f"Refresh token creation too slow: {avg_refresh_time:.4f}s"
        
        print(f"Token creation - Access: {avg_access_time:.4f}s, Refresh: {avg_refresh_time:.4f}s")
    
    def test_jwt_token_verification_performance(self, jwt_handler):
        """Test JWT token verification performance."""
        user_data = {
            "sub": "user_123",
            "username": "perfuser",
            "email": "perf@example.com",
            "roles": ["user"]
        }
        
        # Create tokens for verification testing
        tokens = []
        for _ in range(100):
            token = jwt_handler.create_access_token(user_data)
            tokens.append(token)
        
        # Test token verification
        verify_times = []
        for token in tokens:
            _, duration = self.measure_time(jwt_handler.verify_token, token)
            verify_times.append(duration)
        
        avg_verify_time = mean(verify_times)
        max_verify_time = max(verify_times)
        
        assert avg_verify_time < 0.01, f"Token verification too slow: {avg_verify_time:.4f}s"
        assert max_verify_time < 0.05, f"Max token verification too slow: {max_verify_time:.4f}s"
        
        print(f"Token verification - Avg: {avg_verify_time:.4f}s, Max: {max_verify_time:.4f}s")
    
    def test_user_store_operations_performance(self, user_store):
        """Test user store operations performance."""
        # Test user creation performance
        creation_times = []
        usernames = []
        
        for i in range(50):
            username = f"perfuser{i}"
            usernames.append(username)
            
            _, duration = self.measure_time(
                user_store.create_user,
                username=username,
                email=f"perfuser{i}@example.com",
                full_name=f"Perf User {i}",
                password="PerfPassword123!"
            )
            creation_times.append(duration)
        
        # Test user authentication performance
        auth_times = []
        for username in usernames[:20]:  # Test subset for speed
            _, duration = self.measure_time(
                user_store.authenticate_user,
                username,
                "PerfPassword123!"
            )
            auth_times.append(duration)
        
        # Test user lookup performance
        lookup_times = []
        for username in usernames[:20]:
            _, duration = self.measure_time(user_store.get_user, username)
            lookup_times.append(duration)
        
        avg_creation = mean(creation_times)
        avg_auth = mean(auth_times)
        avg_lookup = mean(lookup_times)
        
        assert avg_creation < 0.1, f"User creation too slow: {avg_creation:.3f}s"
        assert avg_auth < 0.1, f"User authentication too slow: {avg_auth:.3f}s"
        assert avg_lookup < 0.01, f"User lookup too slow: {avg_lookup:.4f}s"
        
        print(f"User store - Creation: {avg_creation:.3f}s, Auth: {avg_auth:.3f}s, Lookup: {avg_lookup:.4f}s")
    
    def test_api_key_operations_performance(self, user_store):
        """Test API key operations performance."""
        # Create test user
        user_store.create_user(
            username="apiperfuser",
            email="apiperf@example.com",
            full_name="API Perf User",
            password="APIPerf123!"
        )
        
        # Test API key generation performance
        generation_times = []
        api_keys = []
        
        for i in range(20):
            start_time = time.perf_counter()
            raw_key, api_key = user_store.create_api_key(
                "apiperfuser",
                f"Perf Key {i}",
                scopes=["read", "write"]
            )
            end_time = time.perf_counter()
            
            generation_times.append(end_time - start_time)
            api_keys.append(raw_key)
        
        # Test API key authentication performance
        auth_times = []
        for raw_key in api_keys:
            _, duration = self.measure_time(user_store.authenticate_api_key, raw_key)
            auth_times.append(duration)
        
        avg_generation = mean(generation_times)
        avg_auth = mean(auth_times)
        
        assert avg_generation < 0.05, f"API key generation too slow: {avg_generation:.3f}s"
        assert avg_auth < 0.02, f"API key authentication too slow: {avg_auth:.3f}s"
        
        print(f"API keys - Generation: {avg_generation:.3f}s, Auth: {avg_auth:.3f}s")
    
    def test_concurrent_authentication_performance(self, user_store, jwt_handler):
        """Test performance under concurrent authentication load."""
        # Create test users
        for i in range(10):
            user_store.create_user(
                username=f"concuser{i}",
                email=f"concuser{i}@example.com",
                full_name=f"Concurrent User {i}",
                password="ConcurrentPass123!"
            )
        
        def authenticate_user(user_id):
            """Authenticate a user and measure time."""
            start_time = time.perf_counter()
            
            # Login
            user = user_store.authenticate_user(f"concuser{user_id}", "ConcurrentPass123!")
            if not user:
                return None, 0
            
            # Create token
            user_data = {
                "sub": user.id,
                "username": user.username,
                "email": user.email
            }
            token = jwt_handler.create_access_token(user_data)
            
            # Verify token
            payload = jwt_handler.verify_token(token)
            
            end_time = time.perf_counter()
            return payload, end_time - start_time
        
        # Test with different concurrency levels
        for num_workers in [1, 5, 10]:
            times = []
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Submit authentication tasks
                futures = []
                for i in range(50):
                    future = executor.submit(authenticate_user, i % 10)
                    futures.append(future)
                
                # Collect results
                for future in as_completed(futures):
                    payload, duration = future.result()
                    if payload:  # Successful authentication
                        times.append(duration)
            
            if times:
                avg_time = mean(times)
                max_time = max(times)
                
                print(f"Concurrent auth ({num_workers} workers) - Avg: {avg_time:.3f}s, Max: {max_time:.3f}s")
                
                # Performance should not degrade significantly with concurrency
                assert avg_time < 0.5, f"Concurrent auth too slow with {num_workers} workers: {avg_time:.3f}s"
                assert max_time < 2.0, f"Max concurrent auth time too high: {max_time:.3f}s"
    
    def test_large_dataset_performance(self, user_store):
        """Test performance with large datasets."""
        # Create a larger number of users
        print("Creating large dataset...")
        creation_start = time.perf_counter()
        
        for i in range(200):
            user_store.create_user(
                username=f"largeuser{i}",
                email=f"largeuser{i}@example.com",
                full_name=f"Large User {i}",
                password="LargeDataset123!"
            )
        
        creation_time = time.perf_counter() - creation_start
        print(f"Created 200 users in {creation_time:.2f}s")
        
        # Test random lookups
        import random
        lookup_times = []
        for _ in range(50):
            user_id = random.randint(0, 199)
            _, duration = self.measure_time(user_store.get_user, f"largeuser{user_id}")
            lookup_times.append(duration)
        
        avg_lookup = mean(lookup_times)
        max_lookup = max(lookup_times)
        
        # Lookup performance should not degrade significantly with dataset size
        assert avg_lookup < 0.01, f"Large dataset lookup too slow: {avg_lookup:.4f}s"
        assert max_lookup < 0.05, f"Max large dataset lookup too slow: {max_lookup:.4f}s"
        
        print(f"Large dataset lookup - Avg: {avg_lookup:.4f}s, Max: {max_lookup:.4f}s")
    
    def test_memory_usage_performance(self, jwt_handler, user_store):
        """Test memory usage characteristics."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create many tokens
        user_data = {"sub": "user_123", "username": "memuser", "email": "mem@example.com"}
        tokens = []
        
        for _ in range(1000):
            token = jwt_handler.create_access_token(user_data)
            tokens.append(token)
        
        # Create many users
        for i in range(100):
            user_store.create_user(
                username=f"memuser{i}",
                email=f"memuser{i}@example.com",
                full_name=f"Memory User {i}",
                password="MemoryTest123!"
            )
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Memory usage - Initial: {initial_memory:.1f}MB, Final: {final_memory:.1f}MB, Increase: {memory_increase:.1f}MB")
        
        # Memory increase should be reasonable
        assert memory_increase < 50, f"Memory usage increased too much: {memory_increase:.1f}MB"
    
    def test_token_cleanup_performance(self, jwt_handler):
        """Test token cleanup performance."""
        user_data = {"sub": "user_123", "username": "cleanupuser", "email": "cleanup@example.com"}
        
        # Create many tokens
        for _ in range(1000):
            jwt_handler.create_access_token(user_data)
            jwt_handler.create_refresh_token(user_data)
        
        initial_count = jwt_handler.get_active_token_count()
        
        # Measure cleanup performance
        _, cleanup_duration = self.measure_time(jwt_handler.cleanup_expired_tokens)
        
        final_count = jwt_handler.get_active_token_count()
        
        print(f"Token cleanup - Initial: {initial_count}, Final: {final_count}, Duration: {cleanup_duration:.3f}s")
        
        # Cleanup should be fast even with many tokens
        assert cleanup_duration < 1.0, f"Token cleanup too slow: {cleanup_duration:.3f}s"
    
    def test_api_key_cleanup_performance(self, user_store):
        """Test API key cleanup performance."""
        # Create test user
        user_store.create_user(
            username="cleanupuser",
            email="cleanup@example.com",
            full_name="Cleanup User",
            password="Cleanup123!"
        )
        
        # Create many API keys
        for i in range(100):
            user_store.create_api_key("cleanupuser", f"Cleanup Key {i}")
        
        initial_count = len(user_store.api_keys)
        
        # Measure cleanup performance
        _, cleanup_duration = self.measure_time(user_store.cleanup_expired_keys)
        
        final_count = len(user_store.api_keys)
        
        print(f"API key cleanup - Initial: {initial_count}, Final: {final_count}, Duration: {cleanup_duration:.3f}s")
        
        # Cleanup should be fast
        assert cleanup_duration < 0.5, f"API key cleanup too slow: {cleanup_duration:.3f}s"
    
    def test_file_io_performance(self, temp_dir):
        """Test file I/O performance for persistence."""
        users_file = temp_dir / "io_perf_users.json"
        api_keys_file = temp_dir / "io_perf_keys.json"
        
        # Test initial creation and loading
        store1 = UserStore(str(users_file), str(api_keys_file))
        
        # Add data
        creation_start = time.perf_counter()
        for i in range(50):
            store1.create_user(
                username=f"iouser{i}",
                email=f"iouser{i}@example.com",
                full_name=f"IO User {i}",
                password="IOTest123!"
            )
        creation_time = time.perf_counter() - creation_start
        
        # Test loading existing data
        load_start = time.perf_counter()
        store2 = UserStore(str(users_file), str(api_keys_file))
        load_time = time.perf_counter() - load_start
        
        print(f"File I/O - Creation: {creation_time:.3f}s, Loading: {load_time:.3f}s")
        
        # File operations should be reasonably fast
        assert creation_time < 5.0, f"File creation too slow: {creation_time:.3f}s"
        assert load_time < 1.0, f"File loading too slow: {load_time:.3f}s"
        
        # Verify data integrity
        assert len(store2.users) == len(store1.users)
    
    @pytest.mark.skip(reason="Long running test - enable for detailed performance analysis")
    def test_sustained_load_performance(self, user_store, jwt_handler):
        """Test performance under sustained load."""
        duration = 60  # Run for 1 minute
        
        # Create test users
        for i in range(20):
            user_store.create_user(
                username=f"loaduser{i}",
                email=f"loaduser{i}@example.com", 
                full_name=f"Load User {i}",
                password="LoadTest123!"
            )
        
        def sustained_operations():
            """Perform sustained auth operations."""
            operations = 0
            start_time = time.perf_counter()
            
            while time.perf_counter() - start_time < duration:
                # Random operations
                import random
                user_id = random.randint(0, 19)
                username = f"loaduser{user_id}"
                
                # Authenticate
                user = user_store.authenticate_user(username, "LoadTest123!")
                if user:
                    # Create token
                    user_data = {"sub": user.id, "username": user.username}
                    token = jwt_handler.create_access_token(user_data)
                    
                    # Verify token
                    jwt_handler.verify_token(token)
                    
                    operations += 1
            
            return operations
        
        # Run sustained load
        total_operations = sustained_operations()
        ops_per_second = total_operations / duration
        
        print(f"Sustained load - {total_operations} operations in {duration}s ({ops_per_second:.1f} ops/sec)")
        
        # Should maintain reasonable throughput
        assert ops_per_second > 10, f"Sustained throughput too low: {ops_per_second:.1f} ops/sec"


class TestPasswordUtilityPerformance:
    """Test performance of password utility functions."""
    
    def test_password_generation_performance(self):
        """Test password generation performance."""
        times = []
        for _ in range(100):
            start_time = time.perf_counter()
            generate_password(16)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        avg_time = mean(times)
        max_time = max(times)
        
        assert avg_time < 0.001, f"Password generation too slow: {avg_time:.4f}s"
        assert max_time < 0.01, f"Max password generation too slow: {max_time:.4f}s"
        
        print(f"Password generation - Avg: {avg_time:.4f}s, Max: {max_time:.4f}s")
    
    def test_api_key_generation_performance(self):
        """Test API key generation performance."""
        times = []
        for _ in range(100):
            start_time = time.perf_counter()
            generate_api_key()
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        avg_time = mean(times)
        max_time = max(times)
        
        assert avg_time < 0.001, f"API key generation too slow: {avg_time:.4f}s"
        assert max_time < 0.01, f"Max API key generation too slow: {max_time:.4f}s"
        
        print(f"API key generation - Avg: {avg_time:.4f}s, Max: {max_time:.4f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
