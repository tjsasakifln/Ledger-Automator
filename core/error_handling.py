# core/error_handling.py
"""
Comprehensive error handling system for Ledger Automator
Provides secure error management, logging, and user-friendly error messages
"""

import sys
import traceback
import logging
from typing import Dict, Any, Optional, Type, Callable, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import functools
import streamlit as st
import json

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for classification"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    DATA_PROCESSING = "data_processing"
    FILE_HANDLING = "file_handling"
    ML_MODEL = "ml_model"
    DATABASE = "database"
    NETWORK = "network"
    SYSTEM = "system"
    UNKNOWN = "unknown"

@dataclass
class ErrorInfo:
    """Structured error information"""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    user_message: str
    technical_details: str
    timestamp: datetime
    context: Dict[str, Any]
    stack_trace: Optional[str] = None
    suggested_action: Optional[str] = None

class LedgerAutomatorError(Exception):
    """Base exception for Ledger Automator"""
    
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.UNKNOWN,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM, context: Dict[str, Any] = None):
        super().__init__(message)
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.timestamp = datetime.now()

class AuthenticationError(LedgerAutomatorError):
    """Authentication related errors"""
    
    def __init__(self, message: str, context: Dict[str, Any] = None):
        super().__init__(message, ErrorCategory.AUTHENTICATION, ErrorSeverity.HIGH, context)

class AuthorizationError(LedgerAutomatorError):
    """Authorization related errors"""
    
    def __init__(self, message: str, context: Dict[str, Any] = None):
        super().__init__(message, ErrorCategory.AUTHORIZATION, ErrorSeverity.HIGH, context)

class ValidationError(LedgerAutomatorError):
    """Data validation errors"""
    
    def __init__(self, message: str, context: Dict[str, Any] = None):
        super().__init__(message, ErrorCategory.VALIDATION, ErrorSeverity.MEDIUM, context)

class FileProcessingError(LedgerAutomatorError):
    """File processing errors"""
    
    def __init__(self, message: str, context: Dict[str, Any] = None):
        super().__init__(message, ErrorCategory.FILE_HANDLING, ErrorSeverity.MEDIUM, context)

class MLModelError(LedgerAutomatorError):
    """Machine learning model errors"""
    
    def __init__(self, message: str, context: Dict[str, Any] = None):
        super().__init__(message, ErrorCategory.ML_MODEL, ErrorSeverity.HIGH, context)

class DataProcessingError(LedgerAutomatorError):
    """Data processing errors"""
    
    def __init__(self, message: str, context: Dict[str, Any] = None):
        super().__init__(message, ErrorCategory.DATA_PROCESSING, ErrorSeverity.MEDIUM, context)

class ErrorHandler:
    """Centralized error handling system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_log_file = "logs/errors.json"
        self.setup_logging()
        
        # Error message mappings for user-friendly display
        self.user_messages = {
            ErrorCategory.AUTHENTICATION: {
                'default': "Authentication failed. Please check your credentials.",
                'account_locked': "Your account has been temporarily locked for security.",
                'session_expired': "Your session has expired. Please log in again."
            },
            ErrorCategory.AUTHORIZATION: {
                'default': "You don't have permission to perform this action.",
                'insufficient_role': "Your role doesn't allow this operation."
            },
            ErrorCategory.VALIDATION: {
                'default': "The data provided is invalid. Please check and try again.",
                'invalid_format': "The file format is not supported.",
                'missing_data': "Required information is missing."
            },
            ErrorCategory.FILE_HANDLING: {
                'default': "There was an issue processing your file.",
                'file_too_large': "The file is too large. Please upload a smaller file.",
                'malicious_content': "The file contains potentially harmful content."
            },
            ErrorCategory.ML_MODEL: {
                'default': "There was an issue with the classification model.",
                'model_not_found': "The classification model is not available.",
                'prediction_failed': "Unable to classify transactions at this time."
            },
            ErrorCategory.DATA_PROCESSING: {
                'default': "There was an issue processing your data.",
                'invalid_transaction': "Some transaction data is invalid.",
                'processing_timeout': "Data processing took too long."
            },
            ErrorCategory.SYSTEM: {
                'default': "A system error occurred. Please try again later.",
                'disk_full': "System storage is full.",
                'memory_error': "System is running low on memory."
            }
        }
        
        # Suggested actions for different error types
        self.suggested_actions = {
            ErrorCategory.AUTHENTICATION: "Try logging in again or contact support if the problem persists.",
            ErrorCategory.AUTHORIZATION: "Contact your administrator to request additional permissions.",
            ErrorCategory.VALIDATION: "Please review your data format and ensure all required fields are filled.",
            ErrorCategory.FILE_HANDLING: "Try uploading a different file or check the file format requirements.",
            ErrorCategory.ML_MODEL: "Please try again later or contact support if the issue persists.",
            ErrorCategory.DATA_PROCESSING: "Please check your data format and try uploading again.",
            ErrorCategory.SYSTEM: "Please try again in a few minutes or contact support."
        }
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        # Create logs directory
        import os
        os.makedirs("logs", exist_ok=True)
        
        # Configure logger
        self.logger.setLevel(logging.INFO)
        
        # File handler for error logs
        error_handler = logging.FileHandler("logs/application.log")
        error_handler.setLevel(logging.ERROR)
        
        # Info handler for general logs
        info_handler = logging.FileHandler("logs/info.log")
        info_handler.setLevel(logging.INFO)
        
        # Console handler for development
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        
        # Secure formatter - no sensitive information
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        error_handler.setFormatter(formatter)
        info_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers if not already added
        if not self.logger.handlers:
            self.logger.addHandler(error_handler)
            self.logger.addHandler(info_handler)
            self.logger.addHandler(console_handler)
    
    def _sanitize_log_message(self, message: str) -> str:
        \"\"\"Sanitize log messages to remove sensitive information\"\"\"
        if not message:
            return message
        
        import re
        
        # Remove common sensitive patterns
        sensitive_patterns = [
            (r'password[=:]\s*[\'"]?([^\s\'"]+)', r'password=***'),
            (r'token[=:]\s*[\'"]?([^\s\'"]+)', r'token=***'),
            (r'secret[=:]\s*[\'"]?([^\s\'"]+)', r'secret=***'),
            (r'key[=:]\s*[\'"]?([^\s\'"]+)', r'key=***'),
            (r'auth[=:]\s*[\'"]?([^\s\'"]+)', r'auth=***'),
            (r'Bearer\s+([^\s]+)', r'Bearer ***'),
            (r'\\b\\d{4}[-\\s]?\\d{4}[-\\s]?\\d{4}[-\\s]?\\d{4}\\b', r'****-****-****-****'),  # Credit card
            (r'\\b\\d{3}-\\d{2}-\\d{4}\\b', r'***-**-****'),  # SSN
            (r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b', r'***@***.***'),  # Email
        ]
        
        sanitized = message
        for pattern, replacement in sensitive_patterns:
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
        
        return sanitized
    
    def _sanitize_context(self, context: dict) -> dict:
        \"\"\"Sanitize context dictionary to remove sensitive information\"\"\"
        if not context:
            return context
        
        sanitized = {}
        sensitive_keys = ['password', 'token', 'secret', 'key', 'auth', 'credential']
        
        for key, value in context.items():
            key_lower = key.lower()
            if any(sensitive_key in key_lower for sensitive_key in sensitive_keys):
                sanitized[key] = '***'
            elif isinstance(value, str):
                sanitized[key] = self._sanitize_log_message(value)
            else:
                sanitized[key] = value
        
        return sanitized
    
    def _sanitize_stack_trace(self, stack_trace: str) -> str:
        \"\"\"Sanitize stack trace to remove sensitive information\"\"\"
        if not stack_trace:
            return stack_trace
        
        # Remove file paths that might contain sensitive information
        import re
        sanitized = re.sub(r'File \"[^\"]*\\\\([^\\\\\"]+)\"', r'File \"...\\\\\\1\"', stack_trace)
        
        # Remove sensitive data patterns
        sanitized = self._sanitize_log_message(sanitized)
        
        return sanitized
    
    def generate_error_id(self) -> str:
        """Generate unique error ID"""
        import uuid
        return f"ERR_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    
    def classify_error(self, error: Exception) -> ErrorCategory:
        """Classify error based on type and context"""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Direct mapping for custom exceptions
        if isinstance(error, LedgerAutomatorError):
            return error.category
        
        # Classification rules based on error type and message
        if error_type in ['AuthenticationError', 'LoginError']:
            return ErrorCategory.AUTHENTICATION
        elif error_type in ['PermissionError', 'AuthorizationError']:
            return ErrorCategory.AUTHORIZATION
        elif error_type in ['ValueError', 'ValidationError'] or 'validation' in error_message:
            return ErrorCategory.VALIDATION
        elif error_type in ['FileNotFoundError', 'IOError', 'OSError'] or 'file' in error_message:
            return ErrorCategory.FILE_HANDLING
        elif 'model' in error_message or 'prediction' in error_message or 'sklearn' in error_message:
            return ErrorCategory.ML_MODEL
        elif error_type in ['pd.errors.ParserError', 'pandas'] or 'dataframe' in error_message:
            return ErrorCategory.DATA_PROCESSING
        elif error_type in ['MemoryError', 'SystemError', 'RuntimeError']:
            return ErrorCategory.SYSTEM
        elif 'network' in error_message or 'connection' in error_message:
            return ErrorCategory.NETWORK
        else:
            return ErrorCategory.UNKNOWN
    
    def determine_severity(self, error: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Determine error severity"""
        if isinstance(error, LedgerAutomatorError):
            return error.severity
        
        # Severity rules
        critical_types = ['SystemError', 'MemoryError', 'SecurityError']
        high_severity_categories = [ErrorCategory.AUTHENTICATION, ErrorCategory.AUTHORIZATION, ErrorCategory.ML_MODEL]
        
        if type(error).__name__ in critical_types:
            return ErrorSeverity.CRITICAL
        elif category in high_severity_categories:
            return ErrorSeverity.HIGH
        elif category in [ErrorCategory.VALIDATION, ErrorCategory.FILE_HANDLING]:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def get_user_message(self, category: ErrorCategory, error_context: str = 'default') -> str:
        """Get user-friendly error message"""
        category_messages = self.user_messages.get(category, {})
        return category_messages.get(error_context, category_messages.get('default', 
                                    "An unexpected error occurred. Please try again."))
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None, 
                    show_to_user: bool = True) -> ErrorInfo:
        """Main error handling method"""
        try:
            # Generate error info
            error_id = self.generate_error_id()
            category = self.classify_error(error)
            severity = self.determine_severity(error, category)
            context = context or {}
            
            # Get appropriate messages
            user_message = self.get_user_message(category)
            technical_details = str(error)
            suggested_action = self.suggested_actions.get(category, "Please contact support.")
            
            # Get stack trace for critical errors only (security consideration)
            stack_trace = None
            if severity == ErrorSeverity.CRITICAL:
                stack_trace = traceback.format_exc()
            elif severity == ErrorSeverity.HIGH:
                # For high severity, only include the error type and message
                stack_trace = f"{type(error).__name__}: {str(error)}"
            
            # Create error info
            error_info = ErrorInfo(
                error_id=error_id,
                category=category,
                severity=severity,
                message=str(error),
                user_message=user_message,
                technical_details=technical_details,
                timestamp=datetime.now(),
                context=context,
                stack_trace=stack_trace,
                suggested_action=suggested_action
            )
            
            # Log error
            self.log_error(error_info)
            
            # Show to user if requested
            if show_to_user:
                self.display_error_to_user(error_info)
            
            return error_info
            
        except Exception as handler_error:
            # Fallback error handling
            self.logger.critical(f"Error handler failed: {str(handler_error)}")
            if show_to_user:
                st.error("A critical system error occurred. Please contact support.")
            
            return ErrorInfo(
                error_id="HANDLER_FAILED",
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.CRITICAL,
                message="Error handler failed",
                user_message="A critical system error occurred",
                technical_details=str(handler_error),
                timestamp=datetime.now(),
                context={}
            )
    
    def log_error(self, error_info: ErrorInfo):
        """Log error with appropriate level"""
        log_data = {
            'error_id': error_info.error_id,
            'category': error_info.category.value,
            'severity': error_info.severity.value,
            'message': error_info.message,
            'timestamp': error_info.timestamp.isoformat(),
            'context': error_info.context
        }
        
        # Log with appropriate level - sanitize sensitive information
        safe_message = self._sanitize_log_message(error_info.message)
        safe_context = self._sanitize_context(error_info.context)
        
        if error_info.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(f"CRITICAL ERROR {error_info.error_id}: {safe_message}")
        elif error_info.severity == ErrorSeverity.HIGH:
            self.logger.error(f"HIGH ERROR {error_info.error_id}: {safe_message}")
        elif error_info.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(f"MEDIUM ERROR {error_info.error_id}: {safe_message}")
        else:
            self.logger.info(f"LOW ERROR {error_info.error_id}: {safe_message}")
        
        # Save detailed error log with sanitized data
        self.save_error_details(error_info)
    
    def save_error_details(self, error_info: ErrorInfo):
        """Save detailed error information with sanitization"""
        try:
            import os
            os.makedirs("logs", exist_ok=True)
            
            error_data = {
                'error_id': error_info.error_id,
                'category': error_info.category.value,
                'severity': error_info.severity.value,
                'message': self._sanitize_log_message(error_info.message),
                'user_message': error_info.user_message,
                'technical_details': self._sanitize_log_message(error_info.technical_details),
                'timestamp': error_info.timestamp.isoformat(),
                'context': self._sanitize_context(error_info.context),
                'stack_trace': self._sanitize_stack_trace(error_info.stack_trace),
                'suggested_action': error_info.suggested_action
            }
            
            # Append to errors log file with secure permissions
            with open(self.error_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(error_data) + '\n')
            
            # Set secure file permissions
            os.chmod(self.error_log_file, 0o600)
                
        except Exception as e:
            self.logger.critical(f"Failed to save error details: {str(e)}")
    
    def display_error_to_user(self, error_info: ErrorInfo):
        """Display error to user in Streamlit - no sensitive information"""
        if error_info.severity == ErrorSeverity.CRITICAL:
            st.error(f"🔴 {error_info.user_message}")
            st.error(f"Error ID: {error_info.error_id}")
            st.error("Please contact support immediately.")
        elif error_info.severity == ErrorSeverity.HIGH:
            st.error(f"🔴 {error_info.user_message}")
            st.info(f"Error ID: {error_info.error_id}")
            if error_info.suggested_action:
                st.info(f"💡 {error_info.suggested_action}")
        elif error_info.severity == ErrorSeverity.MEDIUM:
            st.warning(f"⚠️ {error_info.user_message}")
            if error_info.suggested_action:
                st.info(f"💡 {error_info.suggested_action}")
        else:
            st.info(f"ℹ️ {error_info.user_message}")

def error_handler_decorator(category: ErrorCategory = ErrorCategory.UNKNOWN, 
                          show_to_user: bool = True):
    """Decorator for automatic error handling"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    'function': func.__name__,
                    'args': str(args)[:200],  # Limit args length
                    'kwargs': str(kwargs)[:200]
                }
                
                error_handler.handle_error(e, context, show_to_user)
                
                # Re-raise for critical errors
                if error_handler.classify_error(e) == ErrorCategory.SYSTEM:
                    raise
                
                return None
        return wrapper
    return decorator

def safe_execute(func: Callable, *args, default_return=None, **kwargs):
    """Safely execute a function with error handling"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        context = {
            'function': func.__name__ if hasattr(func, '__name__') else str(func),
            'args': str(args)[:200],
            'kwargs': str(kwargs)[:200]
        }
        error_handler.handle_error(e, context, show_to_user=True)
        return default_return

# Global error handler instance
error_handler = ErrorHandler()

# Streamlit error handling integration
def setup_streamlit_error_handling():
    """Setup Streamlit-specific error handling"""
    def handle_streamlit_error(e):
        error_handler.handle_error(e, {'source': 'streamlit'}, show_to_user=True)
    
    # Override Streamlit's default error handling
    if hasattr(st, 'exception'):
        original_exception = st.exception
        
        def custom_exception(exception):
            handle_streamlit_error(exception)
            return original_exception(exception)
        
        st.exception = custom_exception