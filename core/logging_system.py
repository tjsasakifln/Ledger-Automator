# core/logging_system.py
"""
Comprehensive logging system for Ledger Automator
Provides structured logging, audit trails, and monitoring capabilities
"""

import logging
import logging.handlers
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import threading
import uuid
from contextlib import contextmanager

class LogLevel(Enum):
    """Log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class LogCategory(Enum):
    """Log categories for organization"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_PROCESSING = "data_processing"
    FILE_OPERATION = "file_operation"
    ML_OPERATION = "ml_operation"
    USER_ACTION = "user_action"
    SYSTEM = "system"
    SECURITY = "security"
    PERFORMANCE = "performance"
    AUDIT = "audit"

@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: datetime
    level: LogLevel
    category: LogCategory
    message: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    operation_id: Optional[str] = None
    component: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class SecurityAuditLog:
    """Security-focused audit logging"""
    
    def __init__(self, log_file: str = "logs/security_audit.log"):
        self.log_file = log_file
        self.logger = self._setup_security_logger()
    
    def _setup_security_logger(self) -> logging.Logger:
        """Setup dedicated security logger"""
        logger = logging.getLogger('security_audit')
        logger.setLevel(logging.INFO)
        
        # Create secure log directory
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        # File handler with rotation
        handler = logging.handlers.RotatingFileHandler(
            self.log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        
        # Secure formatter
        formatter = logging.Formatter(
            '%(asctime)s - SECURITY - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        if not logger.handlers:
            logger.addHandler(handler)
        
        return logger
    
    def log_login_attempt(self, username: str, success: bool, ip_address: str = None):
        """Log login attempt"""
        self.logger.info(json.dumps({
            'event': 'login_attempt',
            'username': username,
            'success': success,
            'ip_address': ip_address,
            'timestamp': datetime.now().isoformat()
        }))
    
    def log_permission_check(self, user_id: str, resource: str, action: str, allowed: bool):
        """Log permission check"""
        self.logger.info(json.dumps({
            'event': 'permission_check',
            'user_id': user_id,
            'resource': resource,
            'action': action,
            'allowed': allowed,
            'timestamp': datetime.now().isoformat()
        }))
    
    def log_sensitive_operation(self, user_id: str, operation: str, details: Dict[str, Any]):
        """Log sensitive operations"""
        self.logger.warning(json.dumps({
            'event': 'sensitive_operation',
            'user_id': user_id,
            'operation': operation,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }))
    
    def log_security_violation(self, event: str, details: Dict[str, Any], severity: str = "HIGH"):
        """Log security violations"""
        self.logger.error(json.dumps({
            'event': 'security_violation',
            'violation_type': event,
            'severity': severity,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }))

class PerformanceLogger:
    """Performance monitoring and logging"""
    
    def __init__(self, log_file: str = "logs/performance.log"):
        self.log_file = log_file
        self.logger = self._setup_performance_logger()
        self.active_operations = {}
        self.lock = threading.Lock()
    
    def _setup_performance_logger(self) -> logging.Logger:
        """Setup performance logger"""
        logger = logging.getLogger('performance')
        logger.setLevel(logging.INFO)
        
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        handler = logging.handlers.RotatingFileHandler(
            self.log_file,
            maxBytes=50*1024*1024,  # 50MB
            backupCount=3
        )
        
        formatter = logging.Formatter(
            '%(asctime)s - PERFORMANCE - %(message)s'
        )
        handler.setFormatter(formatter)
        
        if not logger.handlers:
            logger.addHandler(handler)
        
        return logger
    
    @contextmanager
    def timer(self, operation_name: str, metadata: Dict[str, Any] = None):
        """Context manager for timing operations"""
        operation_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        with self.lock:
            self.active_operations[operation_id] = {
                'name': operation_name,
                'start_time': start_time,
                'metadata': metadata or {}
            }
        
        try:
            yield operation_id
        finally:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            with self.lock:
                operation_info = self.active_operations.pop(operation_id, {})
            
            self.log_operation_time(
                operation_name, 
                duration, 
                operation_info.get('metadata', {})
            )
    
    def log_operation_time(self, operation: str, duration: float, metadata: Dict[str, Any] = None):
        """Log operation timing"""
        log_data = {
            'operation': operation,
            'duration_seconds': duration,
            'timestamp': datetime.now().isoformat()
        }
        
        if metadata:
            log_data['metadata'] = metadata
        
        # Warn on slow operations
        if duration > 5.0:  # 5 seconds
            self.logger.warning(f"SLOW_OPERATION: {json.dumps(log_data)}")
        else:
            self.logger.info(json.dumps(log_data))
    
    def log_memory_usage(self, component: str, memory_mb: float):
        """Log memory usage"""
        self.logger.info(json.dumps({
            'metric': 'memory_usage',
            'component': component,
            'memory_mb': memory_mb,
            'timestamp': datetime.now().isoformat()
        }))
    
    def log_file_operation(self, operation: str, file_size: int, duration: float):
        """Log file operation performance"""
        throughput = file_size / (duration * 1024 * 1024) if duration > 0 else 0  # MB/s
        
        self.logger.info(json.dumps({
            'metric': 'file_operation',
            'operation': operation,
            'file_size_bytes': file_size,
            'duration_seconds': duration,
            'throughput_mb_per_sec': throughput,
            'timestamp': datetime.now().isoformat()
        }))

class StructuredLogger:
    """Main structured logging system"""
    
    def __init__(self, component_name: str = "ledger_automator"):
        self.component_name = component_name
        self.logger = self._setup_main_logger()
        self.security_logger = SecurityAuditLog()
        self.performance_logger = PerformanceLogger()
        
        # Thread-local storage for context
        self.local = threading.local()
    
    def _setup_main_logger(self) -> logging.Logger:
        """Setup main application logger"""
        logger = logging.getLogger(self.component_name)
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers to avoid duplicates
        logger.handlers.clear()
        
        # Create logs directory
        os.makedirs("logs", exist_ok=True)
        
        # Application log handler
        app_handler = logging.handlers.RotatingFileHandler(
            "logs/application.log",
            maxBytes=20*1024*1024,  # 20MB
            backupCount=5
        )
        app_handler.setLevel(logging.INFO)
        
        # Error log handler
        error_handler = logging.handlers.RotatingFileHandler(
            "logs/error.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        
        # Console handler for development
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.WARNING)
        
        # JSON formatter for structured logging
        class JsonFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                    'level': record.levelname,
                    'component': record.name,
                    'message': record.getMessage(),
                    'module': record.module,
                    'function': record.funcName,
                    'line': record.lineno
                }
                
                # Add context information if available
                if hasattr(record, 'user_id'):
                    log_entry['user_id'] = record.user_id
                if hasattr(record, 'session_id'):
                    log_entry['session_id'] = record.session_id
                if hasattr(record, 'operation_id'):
                    log_entry['operation_id'] = record.operation_id
                if hasattr(record, 'category'):
                    log_entry['category'] = record.category
                if hasattr(record, 'metadata'):
                    log_entry['metadata'] = record.metadata
                
                return json.dumps(log_entry)
        
        # Human-readable formatter for console
        human_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        app_handler.setFormatter(JsonFormatter())
        error_handler.setFormatter(JsonFormatter())
        console_handler.setFormatter(human_formatter)
        
        logger.addHandler(app_handler)
        logger.addHandler(error_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def set_context(self, user_id: str = None, session_id: str = None, operation_id: str = None):
        """Set logging context for current thread"""
        self.local.user_id = user_id
        self.local.session_id = session_id
        self.local.operation_id = operation_id
    
    def clear_context(self):
        """Clear logging context"""
        for attr in ['user_id', 'session_id', 'operation_id']:
            if hasattr(self.local, attr):
                delattr(self.local, attr)
    
    def _add_context(self, extra: Dict[str, Any]) -> Dict[str, Any]:
        """Add context information to log entry"""
        context = extra.copy() if extra else {}
        
        # Add thread-local context
        if hasattr(self.local, 'user_id') and self.local.user_id:
            context['user_id'] = self.local.user_id
        if hasattr(self.local, 'session_id') and self.local.session_id:
            context['session_id'] = self.local.session_id
        if hasattr(self.local, 'operation_id') and self.local.operation_id:
            context['operation_id'] = self.local.operation_id
        
        return context
    
    def log(self, level: LogLevel, category: LogCategory, message: str, 
           metadata: Dict[str, Any] = None, extra: Dict[str, Any] = None):
        """Main logging method"""
        extra = self._add_context(extra or {})
        extra['category'] = category.value
        
        if metadata:
            extra['metadata'] = metadata
        
        getattr(self.logger, level.value.lower())(message, extra=extra)
    
    def debug(self, message: str, category: LogCategory = LogCategory.SYSTEM, 
             metadata: Dict[str, Any] = None):
        """Debug level logging"""
        self.log(LogLevel.DEBUG, category, message, metadata)
    
    def info(self, message: str, category: LogCategory = LogCategory.SYSTEM, 
            metadata: Dict[str, Any] = None):
        """Info level logging"""
        self.log(LogLevel.INFO, category, message, metadata)
    
    def warning(self, message: str, category: LogCategory = LogCategory.SYSTEM, 
               metadata: Dict[str, Any] = None):
        """Warning level logging"""
        self.log(LogLevel.WARNING, category, message, metadata)
    
    def error(self, message: str, category: LogCategory = LogCategory.SYSTEM, 
             metadata: Dict[str, Any] = None):
        """Error level logging"""
        self.log(LogLevel.ERROR, category, message, metadata)
    
    def critical(self, message: str, category: LogCategory = LogCategory.SYSTEM, 
                metadata: Dict[str, Any] = None):
        """Critical level logging"""
        self.log(LogLevel.CRITICAL, category, message, metadata)
    
    # Convenience methods for common operations
    def log_user_action(self, action: str, details: Dict[str, Any] = None):
        """Log user action"""
        self.info(f"User action: {action}", LogCategory.USER_ACTION, details)
    
    def log_data_operation(self, operation: str, record_count: int = None, 
                          details: Dict[str, Any] = None):
        """Log data processing operation"""
        metadata = details or {}
        if record_count is not None:
            metadata['record_count'] = record_count
        
        self.info(f"Data operation: {operation}", LogCategory.DATA_PROCESSING, metadata)
    
    def log_file_operation(self, operation: str, filename: str, size_bytes: int = None):
        """Log file operation"""
        metadata = {'filename': filename}
        if size_bytes is not None:
            metadata['size_bytes'] = size_bytes
        
        self.info(f"File operation: {operation}", LogCategory.FILE_OPERATION, metadata)
    
    def log_ml_operation(self, operation: str, model_info: Dict[str, Any] = None):
        """Log ML operation"""
        self.info(f"ML operation: {operation}", LogCategory.ML_OPERATION, model_info)
    
    def log_authentication_event(self, event: str, username: str, success: bool):
        """Log authentication event"""
        metadata = {'username': username, 'success': success}
        self.info(f"Authentication: {event}", LogCategory.AUTHENTICATION, metadata)
        
        # Also log to security audit
        self.security_logger.log_login_attempt(username, success)
    
    def log_authorization_check(self, resource: str, action: str, allowed: bool):
        """Log authorization check"""
        metadata = {'resource': resource, 'action': action, 'allowed': allowed}
        self.info(f"Authorization check: {action} on {resource}", 
                 LogCategory.AUTHORIZATION, metadata)
        
        # Also log to security audit
        user_id = getattr(self.local, 'user_id', 'unknown')
        self.security_logger.log_permission_check(user_id, resource, action, allowed)
    
    def log_security_event(self, event: str, severity: str = "MEDIUM", 
                          details: Dict[str, Any] = None):
        """Log security event"""
        self.warning(f"Security event: {event}", LogCategory.SECURITY, details)
        
        # Also log to security audit
        self.security_logger.log_security_violation(event, details or {}, severity)
    
    @contextmanager
    def operation_context(self, operation_name: str, metadata: Dict[str, Any] = None):
        """Context manager for tracking operations"""
        operation_id = str(uuid.uuid4())
        old_operation_id = getattr(self.local, 'operation_id', None)
        
        try:
            self.local.operation_id = operation_id
            self.info(f"Starting operation: {operation_name}", 
                     LogCategory.SYSTEM, metadata)
            
            with self.performance_logger.timer(operation_name, metadata) as timer_id:
                yield operation_id
                
            self.info(f"Completed operation: {operation_name}", LogCategory.SYSTEM)
            
        except Exception as e:
            self.error(f"Failed operation: {operation_name} - {str(e)}", 
                      LogCategory.SYSTEM, {'error': str(e)})
            raise
        finally:
            self.local.operation_id = old_operation_id
    
    def get_log_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get logging statistics"""
        try:
            log_files = [
                "logs/application.log",
                "logs/error.log",
                "logs/security_audit.log",
                "logs/performance.log"
            ]
            
            stats = {
                'period_hours': hours,
                'log_files': {},
                'total_entries': 0,
                'error_count': 0,
                'warning_count': 0
            }
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            for log_file in log_files:
                if os.path.exists(log_file):
                    file_stats = {
                        'size_bytes': os.path.getsize(log_file),
                        'entries': 0,
                        'latest_timestamp': None
                    }
                    
                    try:
                        with open(log_file, 'r', encoding='utf-8') as f:
                            for line in f:
                                if line.strip():
                                    file_stats['entries'] += 1
                                    if 'ERROR' in line:
                                        stats['error_count'] += 1
                                    elif 'WARNING' in line:
                                        stats['warning_count'] += 1
                    except Exception:
                        pass  # Skip files that can't be read
                    
                    stats['log_files'][log_file] = file_stats
                    stats['total_entries'] += file_stats['entries']
            
            return stats
            
        except Exception as e:
            self.error(f"Failed to get log stats: {str(e)}")
            return {'error': str(e)}

# Global logger instance
app_logger = StructuredLogger("ledger_automator")

# Convenience functions
def log_user_action(action: str, details: Dict[str, Any] = None):
    """Log user action"""
    app_logger.log_user_action(action, details)

def log_data_operation(operation: str, record_count: int = None, details: Dict[str, Any] = None):
    """Log data operation"""
    app_logger.log_data_operation(operation, record_count, details)

def log_security_event(event: str, severity: str = "MEDIUM", details: Dict[str, Any] = None):
    """Log security event"""
    app_logger.log_security_event(event, severity, details)

def set_logging_context(user_id: str = None, session_id: str = None, operation_id: str = None):
    """Set logging context"""
    app_logger.set_context(user_id, session_id, operation_id)