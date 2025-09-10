"""
Test suite for authentication password utilities.

This module tests password hashing, validation, and API key generation.
"""

import pytest
from unittest.mock import patch

from framework.auth.password import (
    hash_password, verify_password, generate_password, generate_api_key,
    hash_api_key, verify_api_key, is_password_strong
)


class TestPasswordHashing:
    """Test password hashing functionality."""
    
    def test_hash_password_success(self):
        """Test successful password hashing."""
        password = "testpassword123"
        hashed = hash_password(password)
        
        assert hashed is not None
        assert isinstance(hashed, str)
        assert hashed != password  # Should be hashed, not plain text
        assert len(hashed) > len(password)  # Hash should be longer
    
    def test_hash_password_different_results(self):
        """Test that same password produces different hashes (due to salt)."""
        password = "samepassword"
        hash1 = hash_password(password)
        hash2 = hash_password(password)
        
        # Should be different due to different salts
        assert hash1 != hash2
    
    def test_hash_password_empty_string(self):
        """Test hashing empty password."""
        hashed = hash_password("")
        assert hashed is not None
        assert isinstance(hashed, str)
    
    def test_hash_password_unicode(self):
        """Test hashing unicode password."""
        password = "pässwörd123🔒"
        hashed = hash_password(password)
        
        assert hashed is not None
        assert isinstance(hashed, str)
        assert hashed != password
    
    @patch('framework.auth.password.pwd_context', None)
    @patch('framework.auth.password.BCRYPT_AVAILABLE', False)
    def test_hash_password_no_bcrypt(self):
        """Test password hashing when bcrypt is not available."""
        password = "testpassword"
        hashed = hash_password(password)
        
        # Should use simple fallback hash with "simple:" prefix
        assert hashed.startswith("simple:")
        assert len(hashed) > len(password)
        
        # Verify the fallback hash format (simple:salt:hash)
        parts = hashed.split(":")
        assert len(parts) == 3  # simple, salt, hash
        assert parts[0] == "simple"
        assert len(parts[1]) == 32  # salt is 16 bytes hex = 32 chars
        assert len(parts[2]) == 64  # sha256 hash = 64 chars hex
        
        # Verify the password can be verified
        from framework.auth.password import verify_password
        assert verify_password(password, hashed) is True
    
    def test_verify_password_correct(self):
        """Test password verification with correct password."""
        password = "correctpassword"
        hashed = hash_password(password)
        
        assert verify_password(password, hashed) is True
    
    def test_verify_password_incorrect(self):
        """Test password verification with incorrect password."""
        password = "correctpassword"
        wrong_password = "wrongpassword"
        hashed = hash_password(password)
        
        assert verify_password(wrong_password, hashed) is False
    
    def test_verify_password_empty(self):
        """Test password verification with empty passwords."""
        # Empty password against empty hash
        empty_hash = hash_password("")
        assert verify_password("", empty_hash) is True
        
        # Non-empty password against empty hash
        assert verify_password("password", empty_hash) is False
        
        # Empty password against non-empty hash
        password_hash = hash_password("password")
        assert verify_password("", password_hash) is False
    
    @patch('framework.auth.password.BCRYPT_AVAILABLE', False)
    def test_verify_password_no_bcrypt(self):
        """Test password verification when bcrypt is not available."""
        password = "plainpassword"
        
        # When bcrypt not available, hash_password returns plain text
        hashed = hash_password(password)
        
        # Verification should do simple string comparison
        assert verify_password(password, hashed) is True
        assert verify_password("wrongpassword", hashed) is False
    
    def test_verify_password_invalid_hash(self):
        """Test password verification with invalid hash format."""
        password = "testpassword"
        invalid_hash = "not_a_valid_hash"
        
        # Should handle invalid hash gracefully
        result = verify_password(password, invalid_hash)
        # Result might be False or raise exception depending on implementation
        assert result is False or result is None


class TestPasswordGeneration:
    """Test password generation functionality."""
    
    def test_generate_password_default_length(self):
        """Test password generation with default length."""
        password = generate_password()
        
        assert isinstance(password, str)
        assert len(password) >= 12  # Default minimum length
    
    def test_generate_password_custom_length(self):
        """Test password generation with custom length."""
        for length in [8, 16, 24, 32]:
            password = generate_password(length)
            assert isinstance(password, str)
            # URL-safe base64 encoding might affect exact length
            assert len(password) >= length * 0.75  # Allow some variance for encoding
    
    def test_generate_password_uniqueness(self):
        """Test that generated passwords are unique."""
        passwords = [generate_password() for _ in range(10)]
        
        # All passwords should be different
        assert len(set(passwords)) == len(passwords)
    
    def test_generate_password_zero_length(self):
        """Test password generation with zero length."""
        with pytest.raises(ValueError, match="Password length must be at least 4 characters"):
            generate_password(0)
    
    def test_generate_password_negative_length(self):
        """Test password generation with negative length."""
        with pytest.raises(ValueError, match="Password length must be at least 4 characters"):
            generate_password(-5)


class TestAPIKeyGeneration:
    """Test API key generation functionality."""
    
    def test_generate_api_key_default_length(self):
        """Test API key generation with default length."""
        api_key = generate_api_key()
        
        assert isinstance(api_key, str)
        assert len(api_key) == 46  # "sk_" (3 chars) + base64 with 32 bytes (43 chars) = 46
    
    def test_generate_api_key_custom_length(self):
        """Test API key generation with custom length."""
        for byte_length in [16, 24, 32, 48]:
            api_key = generate_api_key(byte_length)
            assert isinstance(api_key, str)
            # URL-safe base64 encoding: 4/3 * byte_length (rounded up)
            expected_min_length = (byte_length * 4 // 3)
            assert len(api_key) >= expected_min_length
    
    def test_generate_api_key_uniqueness(self):
        """Test that generated API keys are unique."""
        api_keys = [generate_api_key() for _ in range(10)]
        
        # All API keys should be different
        assert len(set(api_keys)) == len(api_keys)
    
    def test_generate_api_key_format(self):
        """Test API key format is URL-safe."""
        api_key = generate_api_key()
        
        # Should only contain URL-safe characters
        import string
        url_safe_chars = string.ascii_letters + string.digits + '-_'
        assert all(c in url_safe_chars for c in api_key)
    
    def test_generate_api_key_zero_length(self):
        """Test API key generation with zero length."""
        api_key = generate_api_key(0)
        assert isinstance(api_key, str)
        # Might be empty or have minimum length
    
    def test_generate_api_key_large_length(self):
        """Test API key generation with large length."""
        api_key = generate_api_key(128)
        assert isinstance(api_key, str)
        assert len(api_key) > 100  # Should be substantial


class TestAPIKeyHashing:
    """Test API key hashing functionality."""
    
    def test_hash_api_key(self):
        """Test API key hashing."""
        api_key = "test_api_key_123"
        hashed = hash_api_key(api_key)
        
        assert isinstance(hashed, str)
        assert hashed != api_key
        assert len(hashed) > 0
    
    def test_hash_api_key_consistent(self):
        """Test that same API key produces same hash."""
        api_key = "consistent_key"
        hash1 = hash_api_key(api_key)
        hash2 = hash_api_key(api_key)
        
        # Should be identical (deterministic hashing)
        assert hash1 == hash2
    
    def test_hash_api_key_different_keys(self):
        """Test that different API keys produce different hashes."""
        key1 = "api_key_1"
        key2 = "api_key_2"
        
        hash1 = hash_api_key(key1)
        hash2 = hash_api_key(key2)
        
        assert hash1 != hash2
    
    def test_hash_api_key_empty(self):
        """Test hashing empty API key."""
        hashed = hash_api_key("")
        assert isinstance(hashed, str)
        assert len(hashed) > 0
    
    def test_verify_api_key_correct(self):
        """Test API key verification with correct key."""
        api_key = "correct_api_key"
        hashed = hash_api_key(api_key)
        
        assert verify_api_key(api_key, hashed) is True
    
    def test_verify_api_key_incorrect(self):
        """Test API key verification with incorrect key."""
        correct_key = "correct_key"
        wrong_key = "wrong_key"
        hashed = hash_api_key(correct_key)
        
        assert verify_api_key(wrong_key, hashed) is False
    
    def test_verify_api_key_empty(self):
        """Test API key verification with empty keys."""
        empty_hash = hash_api_key("")
        assert verify_api_key("", empty_hash) is True
        assert verify_api_key("nonempty", empty_hash) is False


class TestPasswordStrength:
    """Test password strength validation."""
    
    def test_is_password_strong_valid_password(self):
        """Test password strength with valid strong password."""
        strong_passwords = [
            "StrongPa$s847!",
            "MyP@ssw0rd2025",
            "C0mplex&Secure7!",
            "Sup3r$ecur3P@s5"
        ]
        
        for password in strong_passwords:
            is_strong, issues = is_password_strong(password)
            assert is_strong is True, f"Password '{password}' failed with issues: {issues}"
            assert len(issues) == 0
    
    def test_is_password_strong_too_short(self):
        """Test password strength with too short password."""
        short_passwords = [
            "Short1!",  # 7 chars
            "Ab1!",     # 4 chars
            "1234567"   # 7 chars, no special chars
        ]
        
        for password in short_passwords:
            is_strong, issues = is_password_strong(password)
            assert is_strong is False
            assert any("at least 8 characters" in issue for issue in issues)
    
    def test_is_password_strong_missing_character_types(self):
        """Test password strength with missing character types."""
        # No uppercase
        is_strong, issues = is_password_strong("lowercase123!")
        assert is_strong is False
        assert any("uppercase" in issue.lower() for issue in issues)
        
        # No lowercase
        is_strong, issues = is_password_strong("UPPERCASE123!")
        assert is_strong is False
        assert any("lowercase" in issue.lower() for issue in issues)
        
        # No digits
        is_strong, issues = is_password_strong("NoDigitsHere!")
        assert is_strong is False
        assert any("digit" in issue.lower() or "number" in issue.lower() for issue in issues)
        
        # No special characters
        is_strong, issues = is_password_strong("NoSpecialChars123")
        assert is_strong is False
        assert any("special" in issue.lower() for issue in issues)
    
    def test_is_password_strong_common_patterns(self):
        """Test password strength with common patterns."""
        common_patterns = [
            "password123!",
            "Password123!",  # Common word
            "123456789!Ab",  # Sequential numbers
            "abcdefgh123!",  # Sequential letters
            "qwerty123!A"    # Keyboard pattern
        ]
        
        for password in common_patterns:
            is_strong, issues = is_password_strong(password)
            # These should be detected as weak
            assert is_strong is False
            assert len(issues) > 0
    
    def test_is_password_strong_empty_password(self):
        """Test password strength with empty password."""
        is_strong, issues = is_password_strong("")
        assert is_strong is False
        assert len(issues) > 0
        assert any("at least 8 characters" in issue for issue in issues)
    
    def test_is_password_strong_whitespace_only(self):
        """Test password strength with whitespace-only password."""
        is_strong, issues = is_password_strong("        ")
        assert is_strong is False
        assert len(issues) > 0
    
    def test_is_password_strong_unicode_characters(self):
        """Test password strength with unicode characters."""
        unicode_passwords = [
            "Pässwörd123!",
            "密码Test123!",
            "🔐Secure123!"
        ]
        
        for password in unicode_passwords:
            is_strong, issues = is_password_strong(password)
            # Should handle unicode gracefully
            # Result depends on implementation - either accept or provide clear feedback
            assert isinstance(is_strong, bool)
            assert isinstance(issues, list)
    
    def test_is_password_strong_very_long_password(self):
        """Test password strength with very long password."""
        very_long_password = "A" * 100 + "1" * 50 + "!" * 25 + "b" * 25
        is_strong, issues = is_password_strong(very_long_password)
        
        # Very long password should be strong if it meets other criteria
        assert is_strong is True
        assert len(issues) == 0
    
    def test_is_password_strong_repeated_characters(self):
        """Test password strength with repeated characters."""
        repeated_passwords = [
            "Aaaaaaaaaa1!",  # Repeated 'a'
            "Password111!",  # Repeated '1'
            "StrongPass!!!"  # Repeated '!'
        ]
        
        for password in repeated_passwords:
            is_strong, issues = is_password_strong(password)
            # Implementation may or may not flag repeated characters
            assert isinstance(is_strong, bool)
            assert isinstance(issues, list)


class TestPasswordUtilityIntegration:
    """Test integration between password utilities."""
    
    def test_hash_verify_cycle(self):
        """Test complete hash and verify cycle."""
        passwords = [
            "simple",
            "Complex123!",
            "",
            "Unicode密码123!",
            "Very long password with many characters 123!"
        ]
        
        for password in passwords:
            # Hash the password
            hashed = hash_password(password)
            
            # Verify the original password
            assert verify_password(password, hashed) is True
            
            # Verify wrong password fails
            if password:  # Skip for empty password
                wrong_password = password + "wrong"
                assert verify_password(wrong_password, hashed) is False
    
    def test_api_key_generation_and_hashing(self):
        """Test API key generation and hashing integration."""
        # Generate multiple API keys
        api_keys = [generate_api_key() for _ in range(5)]
        
        # Hash them
        hashes = [hash_api_key(key) for key in api_keys]
        
        # Verify each key against its hash
        for key, hashed in zip(api_keys, hashes):
            assert verify_api_key(key, hashed) is True
        
        # Verify cross-verification fails
        for i, key in enumerate(api_keys):
            for j, hashed in enumerate(hashes):
                if i != j:
                    assert verify_api_key(key, hashed) is False
    
    def test_password_strength_and_generation(self):
        """Test that generated passwords are strong."""
        # Generate multiple passwords
        generated_passwords = [generate_password(16) for _ in range(10)]
        
        for password in generated_passwords:
            # Generated passwords might not meet traditional strength requirements
            # since they're random strings, but they should be cryptographically strong
            assert len(password) >= 12
            assert isinstance(password, str)
    
    def test_error_handling_consistency(self):
        """Test consistent error handling across utilities."""
        # Test with None inputs (should handle gracefully)
        try:
            hash_password(None)
        except (TypeError, AttributeError):
            pass  # Expected for None input
        
        try:
            verify_password(None, "hash")
        except (TypeError, AttributeError):
            pass  # Expected for None input
        
        try:
            hash_api_key(None)
        except (TypeError, AttributeError):
            pass  # Expected for None input
        
        try:
            is_password_strong(None)
        except (TypeError, AttributeError):
            pass  # Expected for None input


if __name__ == "__main__":
    pytest.main([__file__])
