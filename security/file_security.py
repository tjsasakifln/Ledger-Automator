# security/file_security.py
"""
Secure file upload and validation system for Ledger Automator
Protects against malicious files, validates content, and ensures data integrity
"""

import streamlit as st
import pandas as pd
import os
import tempfile
import hashlib
import mimetypes
import magic
import logging
import re
from typing import Tuple, Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta
import io
import csv

class SecurityError(Exception):
    """Custom security exception for file handling"""
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FileValidationResult:
    """Result of file validation"""
    is_valid: bool
    file_hash: str
    warnings: List[str]
    errors: List[str]
    metadata: Dict[str, Any]

class SecureFileUpload:
    """Secure file upload and validation system"""
    
    def __init__(self):
        # Security settings
        self.MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
        self.MAX_ROWS = 50000  # Maximum rows in CSV
        self.MAX_COLUMNS = 20  # Maximum columns in CSV
        self.ALLOWED_EXTENSIONS = {'.csv'}
        self.ALLOWED_MIME_TYPES = {'text/csv', 'application/csv', 'text/plain'}
        
        # Quarantine directory with secure path validation
        self.quarantine_dir = Path("security/quarantine")
        # Validate path to prevent directory traversal
        if not str(self.quarantine_dir.resolve()).startswith(str(Path.cwd().resolve())):
            raise SecurityError("Invalid quarantine directory path")
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        # Set secure permissions
        import stat
        os.chmod(self.quarantine_dir, stat.S_IRWXU)  # 700 permissions
        
        # Virus scanning patterns (basic)
        self.malicious_patterns = [
            rb'<script[^>]*>',
            rb'javascript:',
            rb'vbscript:',
            rb'data:text/html',
            rb'=cmd\|',
            rb'=EXEC\(',
            rb'=SYSTEM\(',
            rb'@import',
            rb'<?php',
            rb'<\?xml'
        ]
        
        # CSV injection patterns - comprehensive list
        self.csv_injection_patterns = [
            r'^[@=+\-\t\r\n]',  # Formulas starting with dangerous chars including whitespace
            r'cmd\s*\|',   # Command execution
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
            r'\\\\',  # UNC paths
            r'\$\{',    # Template injection
            r'<%',      # Server-side template injection
            r'\{\{',    # Mustache/Handlebars injection
        ]
    
    def validate_file_upload(self, uploaded_file) -> FileValidationResult:
        """Comprehensive file validation"""
        warnings = []
        errors = []
        metadata = {}
        
        try:
            # Basic file properties
            filename = uploaded_file.name
            file_size = uploaded_file.size
            
            metadata.update({
                'filename': filename,
                'size': file_size,
                'upload_time': datetime.now().isoformat()
            })
            
            # 1. File size validation
            if file_size > self.MAX_FILE_SIZE:
                errors.append(f"File too large: {file_size / (1024*1024):.1f}MB (max: {self.MAX_FILE_SIZE / (1024*1024)}MB)")
            
            if file_size == 0:
                errors.append("File is empty")
            
            # 2. File extension validation
            file_ext = Path(filename).suffix.lower()
            if file_ext not in self.ALLOWED_EXTENSIONS:
                errors.append(f"Invalid file extension: {file_ext}. Allowed: {', '.join(self.ALLOWED_EXTENSIONS)}")
            
            # 3. Read file content for validation
            content = uploaded_file.read()
            uploaded_file.seek(0)  # Reset file pointer
            
            # 4. Calculate file hash
            file_hash = hashlib.sha256(content).hexdigest()
            metadata['sha256'] = file_hash
            
            # 5. MIME type validation
            mime_type = magic.from_buffer(content, mime=True)
            metadata['mime_type'] = mime_type
            
            if mime_type not in self.ALLOWED_MIME_TYPES:
                errors.append(f"Invalid MIME type: {mime_type}. Expected CSV file.")
            
            # 6. Basic malware scanning
            malware_found = self._scan_for_malware(content)
            if malware_found:
                errors.append("Potentially malicious content detected")
                self._quarantine_file(content, filename, "malware_detection")
            
            # 7. CSV structure validation
            if not errors:  # Only if no critical errors so far
                csv_validation = self._validate_csv_structure(content)
                if not csv_validation['is_valid']:
                    errors.extend(csv_validation['errors'])
                warnings.extend(csv_validation['warnings'])
                metadata.update(csv_validation['metadata'])
            
            # 8. Content injection scanning
            injection_found = self._scan_csv_injection(content)
            if injection_found:
                warnings.append("Potential CSV injection patterns detected and sanitized")
            
            is_valid = len(errors) == 0
            
            return FileValidationResult(
                is_valid=is_valid,
                file_hash=file_hash,
                warnings=warnings,
                errors=errors,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"File validation error: {str(e)}")
            return FileValidationResult(
                is_valid=False,
                file_hash="",
                warnings=[],
                errors=[f"Validation failed: {str(e)}"],
                metadata=metadata
            )
    
    def _scan_for_malware(self, content: bytes) -> bool:
        """Basic malware pattern scanning"""
        try:
            content_lower = content.lower()
            
            for pattern in self.malicious_patterns:
                if re.search(pattern, content_lower):
                    logger.warning(f"Malicious pattern detected: {pattern}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Malware scanning error: {str(e)}")
            return True  # Err on the side of caution
    
    def _validate_csv_structure(self, content: bytes) -> Dict[str, Any]:
        """Validate CSV file structure and content"""
        try:
            # Decode content
            try:
                text_content = content.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    text_content = content.decode('latin-1')
                except UnicodeDecodeError:
                    return {
                        'is_valid': False,
                        'errors': ['File encoding not supported'],
                        'warnings': [],
                        'metadata': {}
                    }
            
            errors = []
            warnings = []
            metadata = {}
            
            # Parse CSV
            csv_reader = csv.reader(io.StringIO(text_content))
            rows = list(csv_reader)
            
            if not rows:
                return {
                    'is_valid': False,
                    'errors': ['CSV file is empty'],
                    'warnings': [],
                    'metadata': {}
                }
            
            # Check row count
            row_count = len(rows)
            metadata['row_count'] = row_count
            
            if row_count > self.MAX_ROWS:
                errors.append(f"Too many rows: {row_count} (max: {self.MAX_ROWS})")
            
            # Check column structure
            headers = rows[0] if rows else []
            column_count = len(headers)
            metadata['column_count'] = column_count
            metadata['headers'] = headers
            
            if column_count > self.MAX_COLUMNS:
                errors.append(f"Too many columns: {column_count} (max: {self.MAX_COLUMNS})")
            
            # Validate required columns for transactions
            required_columns = ['Date', 'Description', 'Amount']
            missing_columns = [col for col in required_columns if col not in headers]
            
            if missing_columns:
                errors.append(f"Missing required columns: {', '.join(missing_columns)}")
            
            # Check for suspicious headers
            suspicious_headers = [h for h in headers if any(
                pattern in h.lower() for pattern in ['script', 'exec', 'cmd', 'eval']
            )]
            if suspicious_headers:
                warnings.append(f"Suspicious column names detected: {', '.join(suspicious_headers)}")
            
            # Validate data types in sample rows
            if len(rows) > 1 and not missing_columns:
                sample_size = min(10, len(rows) - 1)
                data_validation = self._validate_sample_data(rows[1:sample_size + 1], headers)
                warnings.extend(data_validation['warnings'])
                metadata.update(data_validation['metadata'])
            
            return {
                'is_valid': len(errors) == 0,
                'errors': errors,
                'warnings': warnings,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"CSV validation error: {str(e)}")
            return {
                'is_valid': False,
                'errors': [f'CSV parsing failed: {str(e)}'],
                'warnings': [],
                'metadata': {}
            }
    
    def _validate_sample_data(self, sample_rows: List[List[str]], headers: List[str]) -> Dict[str, Any]:
        """Validate sample data from CSV"""
        warnings = []
        metadata = {}
        
        try:
            # Find column indices
            date_idx = headers.index('Date') if 'Date' in headers else -1
            desc_idx = headers.index('Description') if 'Description' in headers else -1
            amount_idx = headers.index('Amount') if 'Amount' in headers else -1
            
            date_errors = 0
            amount_errors = 0
            empty_descriptions = 0
            
            for row in sample_rows:
                if len(row) != len(headers):
                    warnings.append(f"Inconsistent column count in data rows")
                    continue
                
                # Validate dates
                if date_idx >= 0 and date_idx < len(row):
                    try:
                        pd.to_datetime(row[date_idx])
                    except:
                        date_errors += 1
                
                # Validate amounts
                if amount_idx >= 0 and amount_idx < len(row):
                    try:
                        float(row[amount_idx])
                    except:
                        amount_errors += 1
                
                # Check descriptions
                if desc_idx >= 0 and desc_idx < len(row):
                    if not row[desc_idx].strip():
                        empty_descriptions += 1
            
            if date_errors > 0:
                warnings.append(f"Found {date_errors} invalid date formats in sample")
            
            if amount_errors > 0:
                warnings.append(f"Found {amount_errors} invalid amount values in sample")
            
            if empty_descriptions > 0:
                warnings.append(f"Found {empty_descriptions} empty descriptions in sample")
            
            metadata.update({
                'sample_size': len(sample_rows),
                'date_errors': date_errors,
                'amount_errors': amount_errors,
                'empty_descriptions': empty_descriptions
            })
            
            return {
                'warnings': warnings,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Sample data validation error: {str(e)}")
            return {
                'warnings': [f'Sample validation failed: {str(e)}'],
                'metadata': {}
            }
    
    def _scan_csv_injection(self, content: bytes) -> bool:
        """Scan for CSV injection patterns"""
        try:
            text_content = content.decode('utf-8', errors='ignore')
            
            for pattern in self.csv_injection_patterns:
                if re.search(pattern, text_content, re.IGNORECASE | re.MULTILINE):
                    logger.warning(f"CSV injection pattern detected: {pattern}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"CSV injection scanning error: {str(e)}")
            return True  # Err on the side of caution
    
    def sanitize_csv_content(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sanitize CSV content to prevent injection"""
        df_clean = df.copy()
        
        # Sanitize text columns
        for column in df_clean.columns:
            if df_clean[column].dtype == 'object':  # Text columns
                df_clean[column] = df_clean[column].apply(self._sanitize_text_field)
        
        return df_clean
    
    def _sanitize_text_field(self, text: str) -> str:
        """Sanitize individual text field"""
        if pd.isna(text) or not isinstance(text, str):
            return text
        
        # Remove dangerous formula prefixes more comprehensively
        dangerous_prefixes = ['=', '+', '-', '@', '\t', '\r', '\n']
        dangerous_functions = ['CMD', 'EXEC', 'SYSTEM', 'HYPERLINK', 'WEBSERVICE']
        
        text = str(text).strip()
        
        # Check for dangerous prefixes
        if text and text[0] in dangerous_prefixes:
            text = "'" + text
        
        # Check for dangerous functions
        text_upper = text.upper()
        for func in dangerous_functions:
            if func in text_upper:
                text = text.replace(func, f"'{func}")
                text = text.replace(func.lower(), f"'{func.lower()}")
        
        # Remove control characters
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        # Limit length
        if len(text) > 1000:
            text = text[:1000] + "..."
        
        return text
    
    def _quarantine_file(self, content: bytes, filename: str, reason: str):
        """Quarantine suspicious file with secure handling"""
        try:
            import stat
            
            # Sanitize filename to prevent path traversal
            safe_filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            quarantine_filename = f"{timestamp}_{reason}_{safe_filename}"
            quarantine_path = self.quarantine_dir / quarantine_filename
            
            # Validate final path
            if not str(quarantine_path.resolve()).startswith(str(self.quarantine_dir.resolve())):
                raise SecurityError("Invalid quarantine file path")
            
            with open(quarantine_path, 'wb') as f:
                f.write(content)
            
            # Set secure file permissions
            os.chmod(quarantine_path, stat.S_IRUSR | stat.S_IWUSR)  # 600 permissions
            
            # Create metadata file
            metadata = {
                'original_filename': filename,
                'quarantine_reason': reason,
                'quarantine_time': datetime.now().isoformat(),
                'file_size': len(content),
                'file_hash': hashlib.sha256(content).hexdigest()
            }
            
            metadata_path = quarantine_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                import json
                json.dump(metadata, f, indent=2)
            
            # Set secure permissions on metadata file
            os.chmod(metadata_path, stat.S_IRUSR | stat.S_IWUSR)  # 600 permissions
            
            logger.warning(f"File quarantined: {quarantine_filename} (reason: {reason})")
            
        except Exception as e:
            logger.error(f"Quarantine failed: {str(e)}")
    
    def secure_file_uploader(self, label: str = "Upload CSV File", 
                           help_text: str = None) -> Optional[pd.DataFrame]:
        """Secure file uploader component for Streamlit"""
        
        if help_text is None:
            help_text = """
            Upload a CSV file containing transaction data.
            Required columns: Date, Description, Amount
            Maximum file size: 10MB
            """
        
        uploaded_file = st.file_uploader(
            label,
            type=['csv'],
            help=help_text,
            accept_multiple_files=False
        )
        
        if uploaded_file is None:
            return None
        
        # Show file information
        st.info(f"📁 File: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")
        
        # Validate file
        with st.spinner("🔍 Validating file security..."):
            validation_result = self.validate_file_upload(uploaded_file)
        
        # Display validation results
        if validation_result.errors:
            st.error("❌ File validation failed:")
            for error in validation_result.errors:
                st.error(f"• {error}")
            return None
        
        if validation_result.warnings:
            st.warning("⚠️ File validation warnings:")
            for warning in validation_result.warnings:
                st.warning(f"• {warning}")
        
        # Show file metadata
        with st.expander("📊 File Information"):
            metadata = validation_result.metadata
            st.json(metadata)
        
        # Load and sanitize CSV
        try:
            df = pd.read_csv(uploaded_file)
            df_sanitized = self.sanitize_csv_content(df)
            
            st.success("✅ File uploaded and validated successfully!")
            
            # Show data preview
            st.subheader("📋 Data Preview")
            st.dataframe(df_sanitized.head(10), use_container_width=True)
            
            return df_sanitized
            
        except Exception as e:
            st.error(f"❌ Error processing CSV: {str(e)}")
            logger.error(f"CSV processing error: {str(e)}")
            return None
    
    def get_quarantine_report(self) -> Dict[str, Any]:
        """Get report of quarantined files"""
        try:
            quarantine_files = list(self.quarantine_dir.glob("*.json"))
            
            report = {
                'total_quarantined': len(quarantine_files),
                'files': []
            }
            
            for metadata_file in quarantine_files:
                with open(metadata_file, 'r') as f:
                    import json
                    metadata = json.load(f)
                    report['files'].append(metadata)
            
            return report
            
        except Exception as e:
            logger.error(f"Quarantine report error: {str(e)}")
            return {'total_quarantined': 0, 'files': []}

# Global secure file upload instance
secure_uploader = SecureFileUpload()