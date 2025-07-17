# security/input_validation.py
"""
Comprehensive input validation and sanitization for Ledger Automator
Protects against injection attacks, validates financial data, and ensures data integrity
"""

import re
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, date
from decimal import Decimal, InvalidOperation
import logging
from dataclasses import dataclass
from enum import Enum
import html
import unicodedata

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Validation strictness levels"""
    STRICT = "strict"
    MODERATE = "moderate" 
    PERMISSIVE = "permissive"

@dataclass
class ValidationResult:
    """Result of input validation"""
    is_valid: bool
    sanitized_value: Any
    original_value: Any
    warnings: List[str]
    errors: List[str]

class InputValidator:
    """Comprehensive input validation and sanitization"""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.MODERATE):
        self.validation_level = validation_level
        
        # CSV Injection patterns - comprehensive list
        self.csv_injection_patterns = [
            r'^[@=+\-\t\r\n]',  # Formula injection including tab/newline
            r'cmd\s*\|',  # Command injection
            r'powershell',
            r'javascript:',
            r'data:',
            r'<script',
            r'=HYPERLINK',
            r'=WEBSERVICE',
            r'=IMPORTXML',
            r'=IMPORTHTML',
            r'=EXEC\(',
            r'=SYSTEM\(',
            r'=CMD\(',
            r'=COMMAND\(',
            r'=SHELL\(',
            r'file://',
            r'\\\\',  # UNC paths
            r'\$\{',  # Template injection
            r'<%',     # Server-side template injection
            r'\{\{',   # Mustache/Handlebars injection
        ]
        
        # XSS patterns
        self.xss_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'vbscript:',
            r'onload\s*=',
            r'onerror\s*=',
            r'onclick\s*=',
            r'onmouseover\s*=',
            r'<iframe[^>]*>',
            r'<object[^>]*>',
            r'<embed[^>]*>',
            r'<link[^>]*>',
            r'<meta[^>]*>'
        ]
        
        # SQL Injection patterns
        self.sql_injection_patterns = [
            r'union\s+select',
            r'drop\s+table',
            r'delete\s+from',
            r'insert\s+into',
            r'update\s+.*\s+set',
            r'exec\s*\(',
            r'sp_executesql',
            r'xp_cmdshell',
            r'--\s*$',
            r'/\*.*\*/',
            r"'.*'",
            r'".*"'
        ]
        
        # Financial transaction validation rules
        self.financial_rules = {
            'amount': {
                'min_value': -1000000,  # -1M
                'max_value': 1000000,   # 1M
                'decimal_places': 2
            },
            'description': {
                'min_length': 1,
                'max_length': 500,
                'allowed_chars': r'^[a-zA-Z0-9\s\.\-\(\)\/\&\#\@\$\%\*\+\=\:\;\,\!\?\_]+$'
            }
        }
    
    def validate_and_sanitize(self, value: Any, field_type: str, 
                             field_name: str = "") -> ValidationResult:
        """Main validation and sanitization method"""
        
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return ValidationResult(
                is_valid=True,
                sanitized_value=None,
                original_value=value,
                warnings=[],
                errors=[]
            )
        
        # Route to specific validator based on field type
        validators = {
            'text': self._validate_text,
            'description': self._validate_description,
            'amount': self._validate_amount,
            'date': self._validate_date,
            'category': self._validate_category,
            'email': self._validate_email,
            'url': self._validate_url
        }
        
        if field_type not in validators:
            return ValidationResult(
                is_valid=False,
                sanitized_value=value,
                original_value=value,
                warnings=[],
                errors=[f"Unknown field type: {field_type}"]
            )
        
        try:
            return validators[field_type](value, field_name)
        except Exception as e:
            logger.error(f"Validation error for {field_name}: {str(e)}")
            return ValidationResult(
                is_valid=False,
                sanitized_value=value,
                original_value=value,
                warnings=[],
                errors=[f"Validation failed: {str(e)}"]
            )
    
    def _validate_text(self, value: str, field_name: str) -> ValidationResult:
        """Validate general text input"""
        warnings = []
        errors = []
        original_value = value
        
        # Convert to string and basic cleaning
        sanitized = str(value).strip()
        
        # Check for injection patterns
        injection_detected = self._detect_injection_patterns(sanitized)
        if injection_detected['detected']:
            if self.validation_level == ValidationLevel.STRICT:
                errors.append(f"Potentially malicious content detected: {injection_detected['pattern']}")
            else:
                warnings.append(f"Suspicious pattern sanitized: {injection_detected['pattern']}")
                sanitized = self._sanitize_injection_patterns(sanitized)
        
        # Basic HTML escaping
        sanitized = html.escape(sanitized)
        
        # Unicode normalization
        sanitized = unicodedata.normalize('NFKC', sanitized)
        
        # Remove control characters but preserve some whitespace
        sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', sanitized)
        
        # Normalize line endings
        sanitized = re.sub(r'\r\n|\r|\n', ' ', sanitized)
        
        # Length validation
        if len(sanitized) > 1000:
            if self.validation_level == ValidationLevel.STRICT:
                errors.append("Text too long (max 1000 characters)")
            else:
                warnings.append("Text truncated to 1000 characters")
                sanitized = sanitized[:1000]
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_value=sanitized,
            original_value=original_value,
            warnings=warnings,
            errors=errors
        )
    
    def _validate_description(self, value: str, field_name: str) -> ValidationResult:
        """Validate transaction description"""
        warnings = []
        errors = []
        original_value = value
        
        # Start with basic text validation
        text_result = self._validate_text(value, field_name)
        sanitized = text_result.sanitized_value
        warnings.extend(text_result.warnings)
        errors.extend(text_result.errors)
        
        if errors:
            return ValidationResult(
                is_valid=False,
                sanitized_value=sanitized,
                original_value=original_value,
                warnings=warnings,
                errors=errors
            )
        
        # Financial description specific rules
        rules = self.financial_rules['description']
        
        # Length validation
        if len(sanitized) < rules['min_length']:
            errors.append(f"Description too short (min {rules['min_length']} characters)")
        
        if len(sanitized) > rules['max_length']:
            if self.validation_level == ValidationLevel.STRICT:
                errors.append(f"Description too long (max {rules['max_length']} characters)")
            else:
                warnings.append(f"Description truncated to {rules['max_length']} characters")
                sanitized = sanitized[:rules['max_length']]
        
        # Character validation
        if not re.match(rules['allowed_chars'], sanitized):
            if self.validation_level == ValidationLevel.STRICT:
                errors.append("Description contains invalid characters")
            else:
                warnings.append("Invalid characters removed from description")
                sanitized = re.sub(r'[^a-zA-Z0-9\s\.\-\(\)\/\&\#\@\$\%\*\+\=\:\;\,\!\?\_]', '', sanitized)
        
        # Additional financial-specific checks
        if re.search(r'\b(test|example|sample|dummy|fake)\b', sanitized.lower()):
            warnings.append("Description appears to be test data")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_value=sanitized,
            original_value=original_value,
            warnings=warnings,
            errors=errors
        )
    
    def _validate_amount(self, value: Union[str, float, int], field_name: str) -> ValidationResult:
        """Validate financial amount"""
        warnings = []
        errors = []
        original_value = value
        
        try:
            # Convert to Decimal for precise financial calculations
            if isinstance(value, str):
                # Remove currency symbols and whitespace
                cleaned = re.sub(r'[^\d\.\-\+]', '', str(value))
                if not cleaned:
                    errors.append("Amount is empty or invalid")
                    return ValidationResult(False, None, original_value, warnings, errors)
                amount = Decimal(cleaned)
            else:
                amount = Decimal(str(value))
            
            # Validation rules
            rules = self.financial_rules['amount']
            
            # Range validation
            if amount < rules['min_value']:
                errors.append(f"Amount too small (min: {rules['min_value']})")
            
            if amount > rules['max_value']:
                errors.append(f"Amount too large (max: {rules['max_value']})")
            
            # Decimal places validation
            if amount.as_tuple().exponent < -rules['decimal_places']:
                if self.validation_level == ValidationLevel.STRICT:
                    errors.append(f"Too many decimal places (max: {rules['decimal_places']})")
                else:
                    warnings.append(f"Amount rounded to {rules['decimal_places']} decimal places")
                    amount = amount.quantize(Decimal('0.01'))
            
            # Suspicious patterns
            if amount == 0:
                warnings.append("Zero amount detected")
            
            if abs(amount) > 100000:
                warnings.append("Large amount detected - please verify")
            
            # Convert back to float for pandas compatibility
            sanitized_amount = float(amount)
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                sanitized_value=sanitized_amount,
                original_value=original_value,
                warnings=warnings,
                errors=errors
            )
            
        except (InvalidOperation, ValueError, TypeError) as e:
            errors.append(f"Invalid amount format: {str(e)}")
            return ValidationResult(
                is_valid=False,
                sanitized_value=None,
                original_value=original_value,
                warnings=warnings,
                errors=errors
            )
    
    def _validate_date(self, value: Union[str, datetime, date], field_name: str) -> ValidationResult:
        """Validate date input"""
        warnings = []
        errors = []
        original_value = value
        
        try:
            # Convert to datetime
            if isinstance(value, str):
                # Try common date formats
                date_formats = [
                    '%Y-%m-%d',
                    '%d/%m/%Y',
                    '%m/%d/%Y',
                    '%Y-%m-%d %H:%M:%S',
                    '%d-%m-%Y',
                    '%Y/%m/%d'
                ]
                
                parsed_date = None
                for fmt in date_formats:
                    try:
                        parsed_date = datetime.strptime(value.strip(), fmt)
                        break
                    except ValueError:
                        continue
                
                if parsed_date is None:
                    # Try pandas parsing as fallback
                    parsed_date = pd.to_datetime(value)
                    
            elif isinstance(value, datetime):
                parsed_date = value
            elif isinstance(value, date):
                parsed_date = datetime.combine(value, datetime.min.time())
            else:
                errors.append("Invalid date type")
                return ValidationResult(False, None, original_value, warnings, errors)
            
            # Validation rules
            current_date = datetime.now()
            min_date = datetime(1900, 1, 1)
            max_date = current_date + pd.DateOffset(years=1)  # Allow 1 year in future
            
            if parsed_date < min_date:
                errors.append(f"Date too old (min: {min_date.strftime('%Y-%m-%d')})")
            
            if parsed_date > max_date:
                errors.append(f"Date too far in future (max: {max_date.strftime('%Y-%m-%d')})")
            
            # Warnings for unusual dates
            if parsed_date > current_date:
                warnings.append("Future date detected")
            
            days_old = (current_date - parsed_date).days
            if days_old > 3650:  # 10 years
                warnings.append("Very old transaction date")
            
            # Return as pandas datetime
            sanitized_date = pd.to_datetime(parsed_date)
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                sanitized_value=sanitized_date,
                original_value=original_value,
                warnings=warnings,
                errors=errors
            )
            
        except Exception as e:
            errors.append(f"Date parsing failed: {str(e)}")
            return ValidationResult(
                is_valid=False,
                sanitized_value=None,
                original_value=original_value,
                warnings=warnings,
                errors=errors
            )
    
    def _validate_category(self, value: str, field_name: str) -> ValidationResult:
        """Validate transaction category"""
        warnings = []
        errors = []
        original_value = value
        
        # Start with text validation
        text_result = self._validate_text(value, field_name)
        sanitized = text_result.sanitized_value
        warnings.extend(text_result.warnings)
        errors.extend(text_result.errors)
        
        if errors:
            return ValidationResult(False, sanitized, original_value, warnings, errors)
        
        # Valid categories (from config.py)
        valid_categories = [
            "Food", "Transportation", "Income", "Healthcare",
            "Utilities", "Entertainment", "Housing", "Shopping", "Other"
        ]
        
        # Case-insensitive matching
        sanitized = sanitized.title()  # Proper case
        
        if sanitized not in valid_categories:
            if self.validation_level == ValidationLevel.STRICT:
                errors.append(f"Invalid category: {sanitized}. Valid: {', '.join(valid_categories)}")
            else:
                # Try to match similar category
                matched_category = self._find_similar_category(sanitized, valid_categories)
                if matched_category:
                    warnings.append(f"Category '{sanitized}' mapped to '{matched_category}'")
                    sanitized = matched_category
                else:
                    warnings.append(f"Unknown category '{sanitized}' mapped to 'Other'")
                    sanitized = "Other"
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_value=sanitized,
            original_value=original_value,
            warnings=warnings,
            errors=errors
        )
    
    def _validate_email(self, value: str, field_name: str) -> ValidationResult:
        """Validate email address"""
        warnings = []
        errors = []
        original_value = value
        
        # Start with text validation
        text_result = self._validate_text(value, field_name)
        sanitized = text_result.sanitized_value
        warnings.extend(text_result.warnings)
        errors.extend(text_result.errors)
        
        if errors:
            return ValidationResult(False, sanitized, original_value, warnings, errors)
        
        # Email regex pattern
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if not re.match(email_pattern, sanitized):
            errors.append("Invalid email format")
        
        # Length validation
        if len(sanitized) > 254:  # RFC 5321 limit
            errors.append("Email address too long")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_value=sanitized.lower(),  # Normalize to lowercase
            original_value=original_value,
            warnings=warnings,
            errors=errors
        )
    
    def _validate_url(self, value: str, field_name: str) -> ValidationResult:
        """Validate URL"""
        warnings = []
        errors = []
        original_value = value
        
        # Start with text validation
        text_result = self._validate_text(value, field_name)
        sanitized = text_result.sanitized_value
        warnings.extend(text_result.warnings)
        errors.extend(text_result.errors)
        
        if errors:
            return ValidationResult(False, sanitized, original_value, warnings, errors)
        
        # URL validation
        url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        
        if not re.match(url_pattern, sanitized, re.IGNORECASE):
            errors.append("Invalid URL format")
        
        # Security checks
        if not sanitized.startswith(('http://', 'https://')):
            errors.append("URL must use HTTP or HTTPS protocol")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_value=sanitized,
            original_value=original_value,
            warnings=warnings,
            errors=errors
        )
    
    def _detect_injection_patterns(self, text: str) -> Dict[str, Any]:
        """Detect various injection patterns"""
        text_lower = text.lower()
        
        # Check CSV injection
        for pattern in self.csv_injection_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return {'detected': True, 'type': 'csv_injection', 'pattern': pattern}
        
        # Check XSS
        for pattern in self.xss_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return {'detected': True, 'type': 'xss', 'pattern': pattern}
        
        # Check SQL injection
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return {'detected': True, 'type': 'sql_injection', 'pattern': pattern}
        
        return {'detected': False, 'type': None, 'pattern': None}
    
    def _sanitize_injection_patterns(self, text: str) -> str:
        """Sanitize known injection patterns"""
        sanitized = text
        
        # Escape dangerous formula prefixes more comprehensively
        dangerous_prefixes = ['@', '=', '+', '-', '\t', '\r', '\n']
        for prefix in dangerous_prefixes:
            if sanitized.startswith(prefix):
                sanitized = "'" + sanitized
                break
        
        # Remove script tags
        sanitized = re.sub(r'<script[^>]*>.*?</script>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove javascript: and data: schemes
        sanitized = re.sub(r'(javascript|data):', '', sanitized, flags=re.IGNORECASE)
        
        # Remove SQL comment patterns and other dangerous constructs
        sanitized = re.sub(r'--.*$', '', sanitized, flags=re.MULTILINE)
        sanitized = re.sub(r'/\*.*?\*/', '', sanitized, flags=re.DOTALL)
        sanitized = re.sub(r';\s*$', '', sanitized)  # Remove trailing semicolons
        sanitized = re.sub(r'\bunion\b.*\bselect\b', '', sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r'\bdrop\b.*\btable\b', '', sanitized, flags=re.IGNORECASE)
        
        return sanitized
    
    def _find_similar_category(self, category: str, valid_categories: List[str]) -> Optional[str]:
        """Find similar category using simple string matching"""
        category_lower = category.lower()
        
        # Direct substring matching
        for valid_cat in valid_categories:
            if category_lower in valid_cat.lower() or valid_cat.lower() in category_lower:
                return valid_cat
        
        # Common aliases
        aliases = {
            'grocery': 'Food',
            'restaurant': 'Food',
            'gas': 'Transportation',
            'fuel': 'Transportation',
            'car': 'Transportation',
            'uber': 'Transportation',
            'salary': 'Income',
            'wage': 'Income',
            'payment': 'Income',
            'doctor': 'Healthcare',
            'hospital': 'Healthcare',
            'medicine': 'Healthcare',
            'electric': 'Utilities',
            'water': 'Utilities',
            'internet': 'Utilities',
            'movie': 'Entertainment',
            'game': 'Entertainment',
            'rent': 'Housing',
            'mortgage': 'Housing',
            'store': 'Shopping',
            'amazon': 'Shopping'
        }
        
        for alias, mapped_category in aliases.items():
            if alias in category_lower:
                return mapped_category
        
        return None
    
    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate entire DataFrame"""
        results = {
            'is_valid': True,
            'total_rows': len(df),
            'valid_rows': 0,
            'warnings': [],
            'errors': [],
            'row_results': [],
            'sanitized_df': df.copy()
        }
        
        # Define column mappings
        column_mappings = {
            'Date': 'date',
            'Description': 'description',
            'Amount': 'amount',
            'Category': 'category'
        }
        
        # Validate each row
        for idx, row in df.iterrows():
            row_valid = True
            row_warnings = []
            row_errors = []
            
            # Validate each column
            for col_name, field_type in column_mappings.items():
                if col_name in df.columns:
                    value = row[col_name]
                    validation_result = self.validate_and_sanitize(
                        value, field_type, f"Row {idx}, {col_name}"
                    )
                    
                    # Update sanitized value
                    results['sanitized_df'].at[idx, col_name] = validation_result.sanitized_value
                    
                    # Collect warnings and errors
                    row_warnings.extend(validation_result.warnings)
                    row_errors.extend(validation_result.errors)
                    
                    if not validation_result.is_valid:
                        row_valid = False
            
            # Store row results
            results['row_results'].append({
                'row_index': idx,
                'is_valid': row_valid,
                'warnings': row_warnings,
                'errors': row_errors
            })
            
            if row_valid:
                results['valid_rows'] += 1
            
            # Add to global warnings/errors
            results['warnings'].extend(row_warnings)
            results['errors'].extend(row_errors)
        
        # Overall validity
        results['is_valid'] = results['valid_rows'] == results['total_rows']
        
        return results

# Global validator instance
input_validator = InputValidator(ValidationLevel.MODERATE)