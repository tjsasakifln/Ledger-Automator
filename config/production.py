# config/production.py
"""
Production configuration for Ledger Automator
Secure, scalable settings for production deployment
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import secrets

@dataclass
class SecurityConfig:
    """Security configuration settings"""
    # Authentication
    SECRET_KEY: str = os.getenv('LEDGER_SECRET_KEY', secrets.token_hex(32))
    JWT_SECRET: str = os.getenv('LEDGER_JWT_SECRET', secrets.token_hex(32))
    JWT_ALGORITHM: str = 'HS256'
    JWT_EXPIRATION_HOURS: int = int(os.getenv('LEDGER_JWT_EXPIRATION', '1'))
    
    # Password Security
    PASSWORD_MIN_LENGTH: int = 12
    PASSWORD_REQUIRE_UPPERCASE: bool = True
    PASSWORD_REQUIRE_LOWERCASE: bool = True
    PASSWORD_REQUIRE_NUMBERS: bool = True
    PASSWORD_REQUIRE_SPECIAL: bool = True
    PASSWORD_HASH_ROUNDS: int = 12
    
    # Session Security
    SESSION_TIMEOUT_MINUTES: int = int(os.getenv('LEDGER_SESSION_TIMEOUT', '60'))
    MAX_LOGIN_ATTEMPTS: int = int(os.getenv('LEDGER_MAX_LOGIN_ATTEMPTS', '5'))
    LOCKOUT_DURATION_MINUTES: int = int(os.getenv('LEDGER_LOCKOUT_DURATION', '15'))
    
    # File Upload Security
    MAX_FILE_SIZE_MB: int = int(os.getenv('LEDGER_MAX_FILE_SIZE', '10'))
    ALLOWED_FILE_TYPES: List[str] = ['.csv']
    ALLOWED_MIME_TYPES: List[str] = ['text/csv', 'application/csv']
    ENABLE_VIRUS_SCANNING: bool = os.getenv('LEDGER_ENABLE_VIRUS_SCAN', 'true').lower() == 'true'
    
    # API Security
    ENABLE_RATE_LIMITING: bool = True
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv('LEDGER_RATE_LIMIT', '60'))
    ENABLE_CORS: bool = False
    ALLOWED_ORIGINS: List[str] = os.getenv('LEDGER_ALLOWED_ORIGINS', '').split(',') if os.getenv('LEDGER_ALLOWED_ORIGINS') else []
    
    # Security Headers
    ENABLE_SECURITY_HEADERS: bool = True
    FORCE_HTTPS: bool = os.getenv('LEDGER_FORCE_HTTPS', 'true').lower() == 'true'
    HSTS_MAX_AGE: int = 31536000  # 1 year

@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    # Connection
    DATABASE_URL: str = os.getenv('DATABASE_URL', 'postgresql://user:pass@localhost:5432/ledger_automator')
    DATABASE_POOL_SIZE: int = int(os.getenv('DB_POOL_SIZE', '20'))
    DATABASE_MAX_OVERFLOW: int = int(os.getenv('DB_MAX_OVERFLOW', '30'))
    DATABASE_POOL_TIMEOUT: int = int(os.getenv('DB_POOL_TIMEOUT', '30'))
    
    # Connection Security
    DATABASE_SSL_MODE: str = os.getenv('DB_SSL_MODE', 'require')
    DATABASE_SSL_CERT: Optional[str] = os.getenv('DB_SSL_CERT')
    DATABASE_SSL_KEY: Optional[str] = os.getenv('DB_SSL_KEY')
    DATABASE_SSL_ROOT_CERT: Optional[str] = os.getenv('DB_SSL_ROOT_CERT')
    
    # Performance
    DATABASE_ECHO: bool = os.getenv('DB_ECHO', 'false').lower() == 'true'
    DATABASE_AUTOCOMMIT: bool = False
    DATABASE_AUTOFLUSH: bool = True
    
    # Backup
    ENABLE_AUTOMATIC_BACKUP: bool = os.getenv('ENABLE_DB_BACKUP', 'true').lower() == 'true'
    BACKUP_INTERVAL_HOURS: int = int(os.getenv('BACKUP_INTERVAL_HOURS', '6'))
    BACKUP_RETENTION_DAYS: int = int(os.getenv('BACKUP_RETENTION_DAYS', '30'))
    BACKUP_S3_BUCKET: Optional[str] = os.getenv('BACKUP_S3_BUCKET')

@dataclass
class CacheConfig:
    """Cache and session storage configuration"""
    # Redis Configuration
    REDIS_URL: str = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    REDIS_PASSWORD: Optional[str] = os.getenv('REDIS_PASSWORD')
    REDIS_SSL: bool = os.getenv('REDIS_SSL', 'false').lower() == 'true'
    
    # Cache Settings
    CACHE_DEFAULT_TIMEOUT: int = int(os.getenv('CACHE_TIMEOUT', '3600'))  # 1 hour
    CACHE_KEY_PREFIX: str = 'ledger_automator:'
    
    # Session Storage
    SESSION_REDIS_DB: int = int(os.getenv('SESSION_REDIS_DB', '1'))
    SESSION_COOKIE_SECURE: bool = os.getenv('SESSION_SECURE', 'true').lower() == 'true'
    SESSION_COOKIE_HTTPONLY: bool = True
    SESSION_COOKIE_SAMESITE: str = 'Lax'

@dataclass
class LoggingConfig:
    """Logging and monitoring configuration"""
    # Log Levels
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    ROOT_LOG_LEVEL: str = os.getenv('ROOT_LOG_LEVEL', 'WARNING')
    
    # Log Destinations
    LOG_TO_FILE: bool = os.getenv('LOG_TO_FILE', 'true').lower() == 'true'
    LOG_TO_STDOUT: bool = os.getenv('LOG_TO_STDOUT', 'true').lower() == 'true'
    LOG_TO_SYSLOG: bool = os.getenv('LOG_TO_SYSLOG', 'false').lower() == 'true'
    
    # Log Files
    LOG_DIR: str = os.getenv('LOG_DIR', '/var/log/ledger-automator')
    LOG_FILE_MAX_SIZE: str = os.getenv('LOG_FILE_MAX_SIZE', '100MB')
    LOG_FILE_BACKUP_COUNT: int = int(os.getenv('LOG_FILE_BACKUP_COUNT', '5'))
    
    # Structured Logging
    LOG_FORMAT: str = 'json'  # json or text
    LOG_INCLUDE_CALLER: bool = True
    LOG_INCLUDE_TIMESTAMP: bool = True
    
    # Security Logging
    SECURITY_LOG_FILE: str = os.path.join(LOG_DIR, 'security.log')
    AUDIT_LOG_FILE: str = os.path.join(LOG_DIR, 'audit.log')
    ENABLE_AUDIT_LOGGING: bool = True
    
    # External Logging
    SENTRY_DSN: Optional[str] = os.getenv('SENTRY_DSN')
    ENABLE_SENTRY: bool = SENTRY_DSN is not None
    SENTRY_ENVIRONMENT: str = os.getenv('SENTRY_ENVIRONMENT', 'production')

@dataclass
class MLConfig:
    """Machine Learning model configuration"""
    # Model Storage
    MODEL_STORAGE_PATH: str = os.getenv('MODEL_STORAGE_PATH', '/var/lib/ledger-automator/models')
    MODEL_BACKUP_PATH: str = os.getenv('MODEL_BACKUP_PATH', '/var/backups/ledger-automator/models')
    
    # Model Performance
    PREDICTION_CONFIDENCE_THRESHOLD: float = float(os.getenv('ML_CONFIDENCE_THRESHOLD', '0.7'))
    MODEL_RETRAIN_THRESHOLD: float = float(os.getenv('ML_RETRAIN_THRESHOLD', '0.1'))  # Accuracy drop
    ENABLE_MODEL_MONITORING: bool = os.getenv('ML_MONITORING', 'true').lower() == 'true'
    
    # Training
    ENABLE_AUTOMATIC_RETRAINING: bool = os.getenv('ML_AUTO_RETRAIN', 'false').lower() == 'true'
    RETRAIN_INTERVAL_DAYS: int = int(os.getenv('ML_RETRAIN_INTERVAL', '30'))
    MIN_TRAINING_SAMPLES: int = int(os.getenv('ML_MIN_TRAINING_SAMPLES', '100'))
    
    # Model Security
    ENABLE_MODEL_SIGNING: bool = os.getenv('ML_MODEL_SIGNING', 'true').lower() == 'true'
    MODEL_SIGNATURE_KEY: str = os.getenv('ML_SIGNATURE_KEY', secrets.token_hex(32))
    
    # Performance Limits
    MAX_PREDICTION_BATCH_SIZE: int = int(os.getenv('ML_MAX_BATCH_SIZE', '1000'))
    PREDICTION_TIMEOUT_SECONDS: int = int(os.getenv('ML_PREDICTION_TIMEOUT', '30'))

@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration"""
    # Health Checks
    ENABLE_HEALTH_CHECKS: bool = True
    HEALTH_CHECK_INTERVAL: int = int(os.getenv('HEALTH_CHECK_INTERVAL', '30'))  # seconds
    
    # Metrics
    ENABLE_PROMETHEUS_METRICS: bool = os.getenv('ENABLE_PROMETHEUS', 'true').lower() == 'true'
    METRICS_PORT: int = int(os.getenv('METRICS_PORT', '9090'))
    METRICS_PATH: str = '/metrics'
    
    # Performance Monitoring
    ENABLE_PERFORMANCE_MONITORING: bool = True
    SLOW_QUERY_THRESHOLD: float = float(os.getenv('SLOW_QUERY_THRESHOLD', '1.0'))  # seconds
    HIGH_MEMORY_THRESHOLD: int = int(os.getenv('HIGH_MEMORY_THRESHOLD', '80'))  # percentage
    
    # Alerting
    ENABLE_EMAIL_ALERTS: bool = os.getenv('ENABLE_EMAIL_ALERTS', 'false').lower() == 'true'
    ALERT_EMAIL_FROM: Optional[str] = os.getenv('ALERT_EMAIL_FROM')
    ALERT_EMAIL_TO: List[str] = os.getenv('ALERT_EMAIL_TO', '').split(',') if os.getenv('ALERT_EMAIL_TO') else []
    SMTP_HOST: Optional[str] = os.getenv('SMTP_HOST')
    SMTP_PORT: int = int(os.getenv('SMTP_PORT', '587'))
    SMTP_USERNAME: Optional[str] = os.getenv('SMTP_USERNAME')
    SMTP_PASSWORD: Optional[str] = os.getenv('SMTP_PASSWORD')
    SMTP_USE_TLS: bool = os.getenv('SMTP_USE_TLS', 'true').lower() == 'true'

@dataclass
class APIConfig:
    """API configuration for REST endpoints"""
    # Server Settings
    HOST: str = os.getenv('API_HOST', '0.0.0.0')
    PORT: int = int(os.getenv('API_PORT', '8000'))
    WORKERS: int = int(os.getenv('API_WORKERS', '4'))
    
    # Performance
    MAX_REQUEST_SIZE: int = int(os.getenv('API_MAX_REQUEST_SIZE', '16777216'))  # 16MB
    REQUEST_TIMEOUT: int = int(os.getenv('API_REQUEST_TIMEOUT', '30'))
    KEEPALIVE_TIMEOUT: int = int(os.getenv('API_KEEPALIVE_TIMEOUT', '5'))
    
    # API Versioning
    API_VERSION: str = 'v1'
    API_PREFIX: str = f'/api/{API_VERSION}'
    
    # Documentation
    ENABLE_API_DOCS: bool = os.getenv('ENABLE_API_DOCS', 'false').lower() == 'true'
    DOCS_PATH: str = '/docs'
    REDOC_PATH: str = '/redoc'

@dataclass
class ProductionConfig:
    """Main production configuration"""
    # Environment
    ENVIRONMENT: str = 'production'
    DEBUG: bool = False
    TESTING: bool = False
    
    # Application
    APP_NAME: str = 'Ledger Automator'
    APP_VERSION: str = os.getenv('APP_VERSION', '1.0.0')
    APP_DESCRIPTION: str = 'Enterprise Financial Transaction Classification System'
    
    # Timezone
    TIMEZONE: str = os.getenv('TZ', 'UTC')
    
    # Feature Flags
    ENABLE_API: bool = os.getenv('ENABLE_API', 'true').lower() == 'true'
    ENABLE_WEB_UI: bool = os.getenv('ENABLE_WEB_UI', 'true').lower() == 'true'
    ENABLE_BATCH_PROCESSING: bool = os.getenv('ENABLE_BATCH_PROCESSING', 'true').lower() == 'true'
    
    # Configuration Components
    security: SecurityConfig = SecurityConfig()
    database: DatabaseConfig = DatabaseConfig()
    cache: CacheConfig = CacheConfig()
    logging: LoggingConfig = LoggingConfig()
    ml: MLConfig = MLConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    api: APIConfig = APIConfig()
    
    def validate_config(self) -> List[str]:
        """Validate production configuration and return any errors"""
        errors = []
        
        # Check required environment variables
        required_vars = [
            'DATABASE_URL',
            'REDIS_URL',
            'LEDGER_SECRET_KEY',
            'LEDGER_JWT_SECRET'
        ]
        
        for var in required_vars:
            if not os.getenv(var):
                errors.append(f"Required environment variable {var} is not set")
        
        # Validate security settings
        if len(self.security.SECRET_KEY) < 32:
            errors.append("SECRET_KEY must be at least 32 characters long")
        
        if len(self.security.JWT_SECRET) < 32:
            errors.append("JWT_SECRET must be at least 32 characters long")
        
        # Validate database configuration
        if not self.database.DATABASE_URL.startswith('postgresql://'):
            errors.append("DATABASE_URL must be a PostgreSQL connection string")
        
        # Validate logging directory
        log_dir = Path(self.logging.LOG_DIR)
        if not log_dir.exists():
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                errors.append(f"Cannot create log directory: {self.logging.LOG_DIR}")
        
        # Validate model storage
        model_dir = Path(self.ml.MODEL_STORAGE_PATH)
        if not model_dir.exists():
            try:
                model_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                errors.append(f"Cannot create model storage directory: {self.ml.MODEL_STORAGE_PATH}")
        
        return errors
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration for SQLAlchemy"""
        config = {
            'url': self.database.DATABASE_URL,
            'pool_size': self.database.DATABASE_POOL_SIZE,
            'max_overflow': self.database.DATABASE_MAX_OVERFLOW,
            'pool_timeout': self.database.DATABASE_POOL_TIMEOUT,
            'echo': self.database.DATABASE_ECHO,
        }
        
        # Add SSL configuration if specified
        if self.database.DATABASE_SSL_MODE:
            ssl_args = {
                'sslmode': self.database.DATABASE_SSL_MODE
            }
            
            if self.database.DATABASE_SSL_CERT:
                ssl_args['sslcert'] = self.database.DATABASE_SSL_CERT
            if self.database.DATABASE_SSL_KEY:
                ssl_args['sslkey'] = self.database.DATABASE_SSL_KEY
            if self.database.DATABASE_SSL_ROOT_CERT:
                ssl_args['sslrootcert'] = self.database.DATABASE_SSL_ROOT_CERT
            
            config['connect_args'] = ssl_args
        
        return config
    
    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration"""
        return {
            'url': self.cache.REDIS_URL,
            'password': self.cache.REDIS_PASSWORD,
            'ssl': self.cache.REDIS_SSL,
            'decode_responses': True,
            'health_check_interval': 30,
            'socket_keepalive': True,
            'socket_keepalive_options': {},
            'retry_on_timeout': True,
            'socket_connect_timeout': 5,
            'socket_timeout': 5
        }
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'json': {
                    'class': 'pythonjsonlogger.jsonlogger.JsonFormatter',
                    'format': '%(asctime)s %(name)s %(levelname)s %(message)s'
                },
                'text': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                }
            },
            'handlers': {
                'file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': os.path.join(self.logging.LOG_DIR, 'application.log'),
                    'maxBytes': self._parse_size(self.logging.LOG_FILE_MAX_SIZE),
                    'backupCount': self.logging.LOG_FILE_BACKUP_COUNT,
                    'formatter': self.logging.LOG_FORMAT,
                    'level': self.logging.LOG_LEVEL
                },
                'console': {
                    'class': 'logging.StreamHandler',
                    'formatter': 'text',
                    'level': self.logging.LOG_LEVEL
                },
                'security': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': self.logging.SECURITY_LOG_FILE,
                    'maxBytes': self._parse_size(self.logging.LOG_FILE_MAX_SIZE),
                    'backupCount': self.logging.LOG_FILE_BACKUP_COUNT,
                    'formatter': 'json',
                    'level': 'INFO'
                }
            },
            'loggers': {
                '': {  # Root logger
                    'handlers': ['file', 'console'],
                    'level': self.logging.ROOT_LOG_LEVEL,
                    'propagate': False
                },
                'ledger_automator': {
                    'handlers': ['file', 'console'],
                    'level': self.logging.LOG_LEVEL,
                    'propagate': False
                },
                'security': {
                    'handlers': ['security'],
                    'level': 'INFO',
                    'propagate': False
                }
            }
        }
    
    def _parse_size(self, size_str: str) -> int:
        """Parse size string like '100MB' to bytes"""
        size_str = size_str.upper()
        if size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)

# Global configuration instance
config = ProductionConfig()

# Validate configuration on import
config_errors = config.validate_config()
if config_errors:
    import sys
    print("Configuration errors found:", file=sys.stderr)
    for error in config_errors:
        print(f"  - {error}", file=sys.stderr)
    if os.getenv('LEDGER_STRICT_CONFIG', 'true').lower() == 'true':
        sys.exit(1)