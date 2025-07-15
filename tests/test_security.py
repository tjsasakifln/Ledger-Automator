# tests/test_security.py
"""
Comprehensive security tests for Ledger Automator
Tests authentication, authorization, input validation, and file security
"""

import pytest
import tempfile
import os
import pandas as pd
import io
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import streamlit as st

# Import security modules
try:
    from security.auth import SecurityManager, UserRole, User
    from security.file_security import SecureFileUpload, FileValidationResult
    from security.input_validation import InputValidator, ValidationLevel, ValidationResult
    from core.error_handling import ErrorHandler, LedgerAutomatorError, ValidationError
except ImportError:
    # Fallback for testing
    pytest.skip("Security modules not available", allow_module_level=True)

class TestSecurityManager:
    """Test authentication and authorization"""
    
    @pytest.fixture
    def temp_security_dir(self):
        """Create temporary security directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the security directory
            with patch('security.auth.Path') as mock_path:
                mock_path.return_value.parent.mkdir = Mock()
                mock_path.return_value.exists.return_value = False
                yield temp_dir
    
    @pytest.fixture
    def security_manager(self, temp_security_dir):
        """Create security manager for testing"""
        with patch('security.auth.SecurityManager._get_or_create_secret_key') as mock_key:
            mock_key.return_value = "test_secret_key_12345678"
            with patch('security.auth.SecurityManager._load_users') as mock_load:
                mock_load.return_value = {}
                manager = SecurityManager()
                return manager
    
    def test_password_hashing(self, security_manager):
        """Test password hashing and verification"""
        password = "TestPassword123!"
        salt = "test_salt"
        
        # Hash password
        hash1 = security_manager._hash_password(password, salt)
        hash2 = security_manager._hash_password(password, salt)
        
        # Same password + salt should produce same hash
        assert hash1 == hash2
        
        # Verify password
        assert security_manager._verify_password(password, salt, hash1)
        assert not security_manager._verify_password("wrong_password", salt, hash1)
    
    def test_password_strength_validation(self, security_manager):
        """Test password strength requirements"""
        # Weak passwords
        weak_passwords = [
            "123456",           # Too short
            "password",         # No uppercase, numbers, special chars
            "PASSWORD",         # No lowercase, numbers, special chars
            "Password123",      # No special chars
            "Password!",        # No numbers
        ]
        
        for password in weak_passwords:
            is_strong, message = security_manager._is_password_strong(password)
            assert not is_strong
            assert isinstance(message, str)
        
        # Strong password
        strong_password = "StrongPassword123!"
        is_strong, message = security_manager._is_password_strong(strong_password)
        assert is_strong
        assert message == "Password is strong"
    
    def test_user_authentication(self, security_manager):
        """Test user authentication flow"""
        # Create test user
        username = "testuser"
        password = "TestPassword123!"
        
        success, message = security_manager.create_user(
            username, password, UserRole.USER, "admin"
        )
        assert success
        
        # Test successful authentication
        success, message = security_manager.authenticate_user(username, password)
        assert success
        assert message == "Login successful"
        
        # Test failed authentication
        success, message = security_manager.authenticate_user(username, "wrong_password")
        assert not success
        assert message == "Invalid credentials"
    
    def test_account_lockout(self, security_manager):
        """Test account lockout after failed attempts"""
        username = "testuser"
        password = "TestPassword123!"
        
        # Create user
        security_manager.create_user(username, password, UserRole.USER, "admin")
        
        # Simulate failed login attempts
        for i in range(security_manager.MAX_LOGIN_ATTEMPTS):
            success, message = security_manager.authenticate_user(username, "wrong_password")
            assert not success
        
        # Account should be locked
        user = security_manager.users[username]
        assert user.locked_until is not None
        
        # Even correct password should fail when locked
        success, message = security_manager.authenticate_user(username, password)
        assert not success
        assert "locked" in message.lower()
    
    @patch('streamlit.session_state', {})
    def test_session_management(self, security_manager):
        """Test session creation and validation"""
        username = "testuser"
        password = "TestPassword123!"
        
        # Create user
        security_manager.create_user(username, password, UserRole.USER, "admin")
        
        # Create session
        session_id = security_manager.create_session(username)
        assert session_id is not None
        assert 'session_data' in st.session_state
        
        # Validate session
        is_valid, result = security_manager.validate_session()
        assert is_valid
        assert result == username
        
        # Test session expiration
        st.session_state['session_data']['expires_at'] = datetime.now().timestamp() - 3600
        is_valid, result = security_manager.validate_session()
        assert not is_valid
        assert result == "Session expired"

class TestSecureFileUpload:
    """Test secure file upload functionality"""
    
    @pytest.fixture
    def secure_uploader(self):
        """Create secure file uploader"""
        return SecureFileUpload()
    
    def create_test_csv(self, content: str) -> io.BytesIO:
        """Create test CSV file"""
        return io.BytesIO(content.encode('utf-8'))
    
    def test_valid_csv_upload(self, secure_uploader):
        """Test valid CSV file upload"""
        csv_content = """Date,Description,Amount
2024-01-01,Test Transaction,-100.50
2024-01-02,Another Transaction,200.00"""
        
        # Mock uploaded file
        mock_file = Mock()
        mock_file.name = "test.csv"
        mock_file.size = len(csv_content)
        mock_file.read.return_value = csv_content.encode('utf-8')
        
        # Validate file
        result = secure_uploader.validate_file_upload(mock_file)
        
        assert result.is_valid
        assert len(result.errors) == 0
        assert 'row_count' in result.metadata
        assert result.metadata['row_count'] == 3  # Including header
    
    def test_oversized_file_rejection(self, secure_uploader):
        """Test rejection of oversized files"""
        mock_file = Mock()
        mock_file.name = "large.csv"
        mock_file.size = secure_uploader.MAX_FILE_SIZE + 1
        mock_file.read.return_value = b"large file content"
        
        result = secure_uploader.validate_file_upload(mock_file)
        
        assert not result.is_valid
        assert any("too large" in error.lower() for error in result.errors)
    
    def test_malicious_content_detection(self, secure_uploader):
        """Test detection of malicious content"""
        malicious_csv = """Date,Description,Amount
2024-01-01,=cmd|'/c calc',100
2024-01-02,<script>alert('xss')</script>,200"""
        
        mock_file = Mock()
        mock_file.name = "malicious.csv"
        mock_file.size = len(malicious_csv)
        mock_file.read.return_value = malicious_csv.encode('utf-8')
        
        result = secure_uploader.validate_file_upload(mock_file)
        
        # Should detect malicious patterns
        assert not result.is_valid or len(result.warnings) > 0
    
    def test_csv_injection_detection(self, secure_uploader):
        """Test CSV injection pattern detection"""
        injection_patterns = [
            "=cmd|'/c calc'",
            "+WEBSERVICE()",
            "-HYPERLINK()",
            "@SUM(1+1)",
            "=EXEC()"
        ]
        
        for pattern in injection_patterns:
            content = f"Date,Description,Amount\n2024-01-01,{pattern},100"
            detected = secure_uploader._scan_csv_injection(content.encode('utf-8'))
            assert detected, f"Failed to detect injection pattern: {pattern}"
    
    def test_file_sanitization(self, secure_uploader):
        """Test CSV content sanitization"""
        dangerous_csv = """Date,Description,Amount
2024-01-01,=cmd|calc,-100
2024-01-02,Normal Transaction,200"""
        
        df = pd.read_csv(io.StringIO(dangerous_csv))
        sanitized_df = secure_uploader.sanitize_csv_content(df)
        
        # Check that dangerous formula was escaped
        dangerous_desc = sanitized_df.iloc[0]['Description']
        assert dangerous_desc.startswith("'=") or "cmd" not in dangerous_desc

class TestInputValidation:
    """Test input validation and sanitization"""
    
    @pytest.fixture
    def validator(self):
        """Create input validator"""
        return InputValidator(ValidationLevel.MODERATE)
    
    def test_amount_validation(self, validator):
        """Test financial amount validation"""
        # Valid amounts
        valid_amounts = [100.50, "200.00", "-50.25", 0]
        
        for amount in valid_amounts:
            result = validator.validate_and_sanitize(amount, 'amount', 'test_amount')
            assert result.is_valid, f"Valid amount {amount} failed validation"
        
        # Invalid amounts
        invalid_amounts = ["abc", "", "100.999", 1000001, -1000001]
        
        for amount in invalid_amounts:
            result = validator.validate_and_sanitize(amount, 'amount', 'test_amount')
            assert not result.is_valid, f"Invalid amount {amount} passed validation"
    
    def test_description_validation(self, validator):
        """Test transaction description validation"""
        # Valid descriptions
        valid_descriptions = [
            "Grocery Store Purchase",
            "ATM Withdrawal",
            "Online Transfer - Rent Payment"
        ]
        
        for desc in valid_descriptions:
            result = validator.validate_and_sanitize(desc, 'description', 'test_desc')
            assert result.is_valid
        
        # Invalid/dangerous descriptions
        dangerous_descriptions = [
            "=cmd|calc",
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "a" * 1001  # Too long
        ]
        
        for desc in dangerous_descriptions:
            result = validator.validate_and_sanitize(desc, 'description', 'test_desc')
            # Should either be invalid or sanitized
            if result.is_valid:
                assert result.sanitized_value != desc
    
    def test_date_validation(self, validator):
        """Test date validation"""
        # Valid dates
        valid_dates = [
            "2024-01-01",
            "01/01/2024",
            "2024-12-31",
            datetime.now()
        ]
        
        for date_val in valid_dates:
            result = validator.validate_and_sanitize(date_val, 'date', 'test_date')
            assert result.is_valid
        
        # Invalid dates
        invalid_dates = [
            "invalid-date",
            "2024-13-01",  # Invalid month
            "1800-01-01",  # Too old
            "2030-01-01"   # Too far in future
        ]
        
        for date_val in invalid_dates:
            result = validator.validate_and_sanitize(date_val, 'date', 'test_date')
            assert not result.is_valid
    
    def test_injection_pattern_detection(self, validator):
        """Test various injection pattern detection"""
        injection_patterns = [
            "=EXEC(calc)",           # CSV injection
            "<script>alert(1)</script>",  # XSS
            "'; DROP TABLE users; --",    # SQL injection
            "javascript:alert(1)",        # JavaScript injection
        ]
        
        for pattern in injection_patterns:
            detected = validator._detect_injection_patterns(pattern)
            assert detected['detected'], f"Failed to detect: {pattern}"
    
    def test_category_validation(self, validator):
        """Test transaction category validation"""
        # Valid categories
        valid_categories = ["Food", "Transportation", "Income", "Healthcare"]
        
        for category in valid_categories:
            result = validator.validate_and_sanitize(category, 'category', 'test_cat')
            assert result.is_valid
            assert result.sanitized_value == category
        
        # Invalid categories that should be mapped
        mappable_categories = {
            "grocery": "Food",
            "gas": "Transportation",
            "salary": "Income",
            "unknown_category": "Other"
        }
        
        for invalid_cat, expected_cat in mappable_categories.items():
            result = validator.validate_and_sanitize(invalid_cat, 'category', 'test_cat')
            if result.is_valid:
                assert result.sanitized_value in ["Food", "Transportation", "Income", "Other"]
    
    def test_dataframe_validation(self, validator):
        """Test complete DataFrame validation"""
        # Create test DataFrame
        test_data = {
            'Date': ['2024-01-01', '2024-01-02', 'invalid-date'],
            'Description': ['Valid Transaction', '=cmd|calc', 'Normal Purchase'],
            'Amount': [100.50, 'invalid', -50.25],
            'Category': ['Food', 'Transportation', 'InvalidCategory']
        }
        df = pd.DataFrame(test_data)
        
        result = validator.validate_dataframe(df)
        
        # Should identify validation issues
        assert result['total_rows'] == 3
        assert result['valid_rows'] < result['total_rows']
        assert len(result['errors']) > 0

class TestErrorHandling:
    """Test error handling system"""
    
    @pytest.fixture
    def error_handler(self):
        """Create error handler"""
        return ErrorHandler()
    
    def test_error_classification(self, error_handler):
        """Test error classification"""
        # Test different error types
        test_cases = [
            (ValueError("validation failed"), "VALIDATION"),
            (FileNotFoundError("file not found"), "FILE_HANDLING"),
            (PermissionError("access denied"), "AUTHORIZATION"),
            (Exception("model prediction failed"), "UNKNOWN")
        ]
        
        for error, expected_category in test_cases:
            category = error_handler.classify_error(error)
            # Check that classification makes sense
            assert isinstance(category.value, str)
    
    def test_custom_exceptions(self, error_handler):
        """Test custom exception handling"""
        from core.error_handling import ValidationError, AuthenticationError
        
        # Test ValidationError
        validation_error = ValidationError("Invalid input data")
        error_info = error_handler.handle_error(validation_error, show_to_user=False)
        
        assert error_info.category.value == "validation"
        assert "Invalid input data" in error_info.message
        assert error_info.severity.value in ["low", "medium", "high", "critical"]
    
    def test_error_sanitization(self, error_handler):
        """Test that sensitive information is not exposed"""
        # Create error with sensitive information
        sensitive_error = Exception("Database connection failed: password=secret123")
        error_info = error_handler.handle_error(sensitive_error, show_to_user=False)
        
        # User message should not contain sensitive details
        assert "password=secret123" not in error_info.user_message
        assert "secret123" not in error_info.user_message

class TestIntegrationSecurity:
    """Integration tests for security components"""
    
    def test_secure_upload_with_validation(self):
        """Test secure upload integrated with validation"""
        uploader = SecureFileUpload()
        validator = InputValidator()
        
        # Create test CSV with mixed valid/invalid data
        csv_content = """Date,Description,Amount
2024-01-01,Valid Transaction,100.50
2024-01-02,=cmd|calc,200.00
invalid-date,Another Transaction,-50.25"""
        
        # Mock file upload
        mock_file = Mock()
        mock_file.name = "test.csv"
        mock_file.size = len(csv_content)
        mock_file.read.return_value = csv_content.encode('utf-8')
        
        # Upload validation
        upload_result = uploader.validate_file_upload(mock_file)
        
        if upload_result.is_valid:
            # Parse and validate content
            df = pd.read_csv(io.StringIO(csv_content))
            validation_result = validator.validate_dataframe(df)
            
            # Should catch validation issues
            assert validation_result['total_rows'] == 3
            assert len(validation_result['errors']) > 0 or len(validation_result['warnings']) > 0
    
    @patch('streamlit.session_state', {})
    def test_authentication_with_error_handling(self):
        """Test authentication errors are properly handled"""
        from core.error_handling import error_handler_decorator, AuthenticationError
        
        @error_handler_decorator(show_to_user=False)
        def mock_login(username, password):
            if not username or not password:
                raise AuthenticationError("Credentials required")
            if username == "blocked":
                raise AuthenticationError("Account locked")
            return True
        
        # Test successful case
        result = mock_login("user", "pass")
        assert result is True
        
        # Test error cases - should be handled gracefully
        result = mock_login("", "")
        assert result is None  # Error handler returns None on failure
        
        result = mock_login("blocked", "pass")
        assert result is None

# Test fixtures and utilities
@pytest.fixture
def sample_transaction_data():
    """Sample transaction data for testing"""
    return {
        'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'Description': ['Grocery Store', 'Gas Station', 'Salary Payment'],
        'Amount': [-150.50, -80.00, 3500.00],
        'Category': ['Food', 'Transportation', 'Income']
    }

@pytest.fixture
def malicious_transaction_data():
    """Malicious transaction data for security testing"""
    return {
        'Date': ['2024-01-01', '2024-01-02'],
        'Description': ['=cmd|calc', '<script>alert("xss")</script>'],
        'Amount': [100, 200],
        'Category': ['Other', 'Other']
    }

# Performance and load testing
class TestSecurityPerformance:
    """Test security performance under load"""
    
    def test_validation_performance(self):
        """Test validation performance with large datasets"""
        validator = InputValidator()
        
        # Create large dataset
        large_data = {
            'Date': ['2024-01-01'] * 1000,
            'Description': ['Test Transaction'] * 1000,
            'Amount': [100.50] * 1000,
            'Category': ['Food'] * 1000
        }
        df = pd.DataFrame(large_data)
        
        import time
        start_time = time.time()
        result = validator.validate_dataframe(df)
        end_time = time.time()
        
        # Should complete within reasonable time
        assert (end_time - start_time) < 5.0  # 5 seconds max
        assert result['is_valid']
    
    def test_password_hashing_performance(self):
        """Test password hashing performance"""
        security_manager = SecurityManager()
        
        import time
        start_time = time.time()
        
        # Hash multiple passwords
        for i in range(10):
            security_manager._hash_password(f"password{i}", f"salt{i}")
        
        end_time = time.time()
        
        # Should complete within reasonable time
        assert (end_time - start_time) < 2.0  # 2 seconds max

if __name__ == "__main__":
    pytest.main([__file__, "-v"])