"""
Custom exception classes for the Ledger Automator application.
Follows Silicon Valley best practices for enterprise error handling.
"""

class LedgerAutomatorError(Exception):
    """Base exception class for all Ledger Automator errors."""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "LEDGER_ERROR"
        self.details = details or {}
    
    def to_dict(self) -> dict:
        """Convert exception to dictionary for logging/API responses."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details
        }

class DataValidationError(LedgerAutomatorError):
    """Raised when input data fails validation checks."""
    
    def __init__(self, message: str, field_name: str = None, invalid_value=None):
        super().__init__(
            message, 
            error_code="DATA_VALIDATION_ERROR",
            details={"field": field_name, "value": str(invalid_value) if invalid_value else None}
        )
        self.field_name = field_name
        self.invalid_value = invalid_value

class FileProcessingError(LedgerAutomatorError):
    """Raised when file operations fail."""
    
    def __init__(self, message: str, file_path: str = None, operation: str = None):
        super().__init__(
            message,
            error_code="FILE_PROCESSING_ERROR", 
            details={"file_path": file_path, "operation": operation}
        )
        self.file_path = file_path
        self.operation = operation

class ModelNotFoundError(LedgerAutomatorError):
    """Raised when ML model files are not found."""
    
    def __init__(self, model_path: str, suggestion: str = None):
        message = f"Model not found at: {model_path}"
        if suggestion:
            message += f". {suggestion}"
        
        super().__init__(
            message,
            error_code="MODEL_NOT_FOUND",
            details={"model_path": model_path, "suggestion": suggestion}
        )
        self.model_path = model_path

class ClassificationError(LedgerAutomatorError):
    """Raised when ML classification fails."""
    
    def __init__(self, message: str, transaction_count: int = None):
        super().__init__(
            message,
            error_code="CLASSIFICATION_ERROR",
            details={"transaction_count": transaction_count}
        )
        self.transaction_count = transaction_count

class InsufficientDataError(LedgerAutomatorError):
    """Raised when there's not enough data for ML operations."""
    
    def __init__(self, required_count: int, actual_count: int, data_type: str = "samples"):
        message = f"Insufficient {data_type}: need at least {required_count}, got {actual_count}"
        super().__init__(
            message,
            error_code="INSUFFICIENT_DATA",
            details={
                "required_count": required_count,
                "actual_count": actual_count,
                "data_type": data_type
            }
        )
        self.required_count = required_count
        self.actual_count = actual_count

class ConfigurationError(LedgerAutomatorError):
    """Raised when configuration is invalid or missing."""
    
    def __init__(self, message: str, config_key: str = None):
        super().__init__(
            message,
            error_code="CONFIGURATION_ERROR",
            details={"config_key": config_key}
        )
        self.config_key = config_key

class SecurityError(LedgerAutomatorError):
    """Raised when security validation fails."""
    
    def __init__(self, message: str, security_check: str = None):
        super().__init__(
            message,
            error_code="SECURITY_ERROR",
            details={"security_check": security_check}
        )
        self.security_check = security_check

# Utility functions for exception handling

def handle_file_operation(func):
    """Decorator to wrap file operations with proper error handling."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            raise FileProcessingError(f"File not found: {str(e)}", operation="read")
        except PermissionError as e:
            raise FileProcessingError(f"Permission denied: {str(e)}", operation="access")
        except OSError as e:
            raise FileProcessingError(f"OS error: {str(e)}", operation="file_system")
        except Exception as e:
            raise FileProcessingError(f"Unexpected file error: {str(e)}", operation="unknown")
    
    return wrapper

def validate_required_columns(df, required_columns: list, context: str = "dataset"):
    """Validate that DataFrame contains required columns."""
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise DataValidationError(
            f"Missing required columns in {context}: {missing_columns}",
            field_name="columns"
        )

def validate_data_types(df, column_types: dict, context: str = "dataset"):
    """Validate DataFrame column data types."""
    for column, expected_type in column_types.items():
        if column in df.columns:
            if not pd.api.types.is_dtype_equal(df[column].dtype, expected_type):
                raise DataValidationError(
                    f"Column '{column}' has incorrect type in {context}. "
                    f"Expected {expected_type}, got {df[column].dtype}",
                    field_name=column
                )

def validate_minimum_rows(df, min_rows: int, context: str = "dataset"):
    """Validate minimum number of rows in DataFrame."""
    if len(df) < min_rows:
        raise InsufficientDataError(
            required_count=min_rows,
            actual_count=len(df),
            data_type=f"{context} rows"
        )