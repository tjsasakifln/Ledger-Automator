# security/auth.py
"""
Comprehensive authentication and authorization system for Ledger Automator
Implements secure user management, session handling, and access control
"""

import streamlit as st
import hashlib
import secrets
import hmac
import time
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path

class SecurityError(Exception):
    """Custom security exception"""
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserRole(Enum):
    """User roles for access control"""
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"

@dataclass
class User:
    """User data model"""
    username: str
    password_hash: str
    salt: str
    role: UserRole
    created_at: float
    last_login: Optional[float] = None
    failed_attempts: int = 0
    locked_until: Optional[float] = None
    is_active: bool = True

class SecurityManager:
    """Main security management class"""
    
    def __init__(self):
        self.users_file = Path("security/users.json")
        self.sessions_file = Path("security/sessions.json")
        self.secret_key = self._get_or_create_secret_key()
        self.users = self._load_users()
        
        # Security settings
        self.MAX_LOGIN_ATTEMPTS = 5
        self.LOCKOUT_DURATION = 900  # 15 minutes
        self.SESSION_TIMEOUT = 3600  # 1 hour
        self.PASSWORD_MIN_LENGTH = 8
        
        # Initialize default admin user if none exists
        if not self.users:
            self._create_default_admin()
    
    def _get_or_create_secret_key(self) -> str:
        """Get or create secret key for session signing"""
        key_file = Path("security/secret.key")
        
        # Validate path to prevent directory traversal
        if not str(key_file.resolve()).startswith(str(Path.cwd().resolve())):
            raise SecurityError("Invalid key file path")
        
        key_file.parent.mkdir(exist_ok=True)
        
        if key_file.exists():
            with open(key_file, 'r') as f:
                return f.read().strip()
        else:
            # Generate new secret key (512-bit for better security)
            key = secrets.token_hex(64)
            with open(key_file, 'w') as f:
                f.write(key)
            # Set secure permissions
            os.chmod(key_file, 0o600)
            return key
    
    def _load_users(self) -> Dict[str, User]:
        """Load users from secure storage"""
        if not self.users_file.exists():
            return {}
        
        try:
            with open(self.users_file, 'r') as f:
                data = json.load(f)
            
            users = {}
            for username, user_data in data.items():
                users[username] = User(
                    username=user_data['username'],
                    password_hash=user_data['password_hash'],
                    salt=user_data['salt'],
                    role=UserRole(user_data['role']),
                    created_at=user_data['created_at'],
                    last_login=user_data.get('last_login'),
                    failed_attempts=user_data.get('failed_attempts', 0),
                    locked_until=user_data.get('locked_until'),
                    is_active=user_data.get('is_active', True)
                )
            return users
        except Exception as e:
            logger.error(f"Error loading users: {e}")
            return {}
    
    def _save_users(self):
        """Save users to secure storage"""
        self.users_file.parent.mkdir(exist_ok=True)
        
        data = {}
        for username, user in self.users.items():
            data[username] = {
                'username': user.username,
                'password_hash': user.password_hash,
                'salt': user.salt,
                'role': user.role.value,
                'created_at': user.created_at,
                'last_login': user.last_login,
                'failed_attempts': user.failed_attempts,
                'locked_until': user.locked_until,
                'is_active': user.is_active
            }
        
        with open(self.users_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Set secure permissions
        os.chmod(self.users_file, 0o600)
    
    def _create_default_admin(self):
        """Create default admin user with secure random password"""
        # Generate secure random password
        default_password = secrets.token_urlsafe(16)
        salt = secrets.token_hex(32)
        password_hash = self._hash_password(default_password, salt)
        
        admin_user = User(
            username="admin",
            password_hash=password_hash,
            salt=salt,
            role=UserRole.ADMIN,
            created_at=time.time(),
            is_active=True
        )
        
        self.users["admin"] = admin_user
        self._save_users()
        
        # Save password to secure file instead of logging
        admin_pwd_file = Path("security/admin_password.txt")
        admin_pwd_file.parent.mkdir(exist_ok=True)
        with open(admin_pwd_file, 'w') as f:
            f.write(f"Default admin password: {default_password}\n")
            f.write("CHANGE THIS PASSWORD IMMEDIATELY!\n")
            f.write("Delete this file after setting new password.\n")
        os.chmod(admin_pwd_file, 0o600)
        
        logger.warning("Default admin created. Password saved to security/admin_password.txt")
        logger.warning("CHANGE THIS PASSWORD IMMEDIATELY!")
    
    def _hash_password(self, password: str, salt: str) -> str:
        """Hash password with salt using PBKDF2"""
        return hashlib.pbkdf2_hex(
            password.encode('utf-8'),
            salt.encode('utf-8'),
            600000,  # increased iterations for better security
            64  # increased key length
        )
    
    def _verify_password(self, password: str, salt: str, hash_expected: str) -> bool:
        """Verify password against hash"""
        hash_actual = self._hash_password(password, salt)
        return hmac.compare_digest(hash_expected, hash_actual)
    
    def _is_password_strong(self, password: str) -> Tuple[bool, str]:
        """Check password strength"""
        if len(password) < self.PASSWORD_MIN_LENGTH:
            return False, f"Password must be at least {self.PASSWORD_MIN_LENGTH} characters"
        
        if not any(c.isupper() for c in password):
            return False, "Password must contain uppercase letters"
        
        if not any(c.islower() for c in password):
            return False, "Password must contain lowercase letters"
        
        if not any(c.isdigit() for c in password):
            return False, "Password must contain numbers"
        
        if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            return False, "Password must contain special characters"
        
        return True, "Password is strong"
    
    def authenticate_user(self, username: str, password: str) -> Tuple[bool, str]:
        """Authenticate user credentials"""
        if username not in self.users:
            logger.warning(f"Login attempt with unknown username: {username}")
            return False, "Invalid credentials"
        
        user = self.users[username]
        current_time = time.time()
        
        # Check if account is locked
        if user.locked_until and current_time < user.locked_until:
            remaining = int(user.locked_until - current_time)
            logger.warning(f"Login attempt on locked account: {username}")
            return False, f"Account locked. Try again in {remaining} seconds"
        
        # Check if account is active
        if not user.is_active:
            logger.warning(f"Login attempt on inactive account: {username}")
            return False, "Account is inactive"
        
        # Verify password
        if not self._verify_password(password, user.salt, user.password_hash):
            user.failed_attempts += 1
            
            # Lock account if too many failed attempts
            if user.failed_attempts >= self.MAX_LOGIN_ATTEMPTS:
                user.locked_until = current_time + self.LOCKOUT_DURATION
                logger.warning(f"Account locked due to failed attempts: {username}")
            
            self._save_users()
            logger.warning(f"Failed login attempt for: {username}")
            return False, "Invalid credentials"
        
        # Successful login
        user.failed_attempts = 0
        user.locked_until = None
        user.last_login = current_time
        self._save_users()
        
        logger.info(f"Successful login: {username}")
        return True, "Login successful"
    
    def create_session(self, username: str) -> str:
        """Create secure session for user"""
        if username not in self.users:
            raise ValueError("User not found")
        
        session_id = secrets.token_urlsafe(48)  # increased token length
        session_data = {
            'username': username,
            'role': self.users[username].role.value,
            'created_at': time.time(),
            'expires_at': time.time() + self.SESSION_TIMEOUT,
            'csrf_token': secrets.token_hex(32)  # Add CSRF token
        }
        
        # Create session signature to prevent tampering
        session_signature = hmac.new(
            self.secret_key.encode(),
            json.dumps(session_data, sort_keys=True).encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Store session in Streamlit session state
        st.session_state['session_id'] = session_id
        st.session_state['session_data'] = session_data
        st.session_state['session_signature'] = session_signature
        st.session_state['authenticated'] = True
        
        logger.info(f"Session created for user: {username}")
        return session_id
    
    def validate_session(self) -> Tuple[bool, Optional[str]]:
        """Validate current session"""
        if 'session_data' not in st.session_state or 'session_signature' not in st.session_state:
            return False, "No session found"
        
        session_data = st.session_state['session_data']
        session_signature = st.session_state['session_signature']
        current_time = time.time()
        
        # Validate session signature to prevent tampering
        expected_signature = hmac.new(
            self.secret_key.encode(),
            json.dumps(session_data, sort_keys=True).encode(),
            hashlib.sha256
        ).hexdigest()
        
        if not hmac.compare_digest(session_signature, expected_signature):
            self.logout()
            return False, "Session tampered"
        
        # Check session expiration
        if current_time > session_data['expires_at']:
            self.logout()
            return False, "Session expired"
        
        # Check if user still exists and is active
        username = session_data['username']
        if username not in self.users or not self.users[username].is_active:
            self.logout()
            return False, "User no longer active"
        
        # Extend session on activity
        session_data['expires_at'] = current_time + self.SESSION_TIMEOUT
        st.session_state['session_data'] = session_data
        
        # Update signature after extending session
        new_signature = hmac.new(
            self.secret_key.encode(),
            json.dumps(session_data, sort_keys=True).encode(),
            hashlib.sha256
        ).hexdigest()
        st.session_state['session_signature'] = new_signature
        
        return True, username
    
    def logout(self):
        """Logout user and clear session"""
        if 'session_data' in st.session_state:
            username = st.session_state['session_data']['username']
            logger.info(f"User logged out: {username}")
        
        # Clear session data
        for key in ['session_id', 'session_data', 'authenticated']:
            if key in st.session_state:
                del st.session_state[key]
    
    def require_auth(self, required_role: UserRole = UserRole.USER) -> Optional[str]:
        """Decorator/middleware to require authentication"""
        is_valid, result = self.validate_session()
        
        if not is_valid:
            self._show_login_form()
            st.stop()
        
        username = result
        user = self.users[username]
        
        # Check role permissions
        role_hierarchy = {
            UserRole.VIEWER: 1,
            UserRole.USER: 2,
            UserRole.ADMIN: 3
        }
        
        if role_hierarchy[user.role] < role_hierarchy[required_role]:
            st.error("❌ Insufficient permissions")
            st.stop()
        
        return username
    
    def _show_login_form(self):
        """Display login form"""
        st.markdown("# 🔐 Ledger Automator - Login")
        st.markdown("Please log in to access the application")
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")
            
            if submit_button:
                if not username or not password:
                    st.error("Please enter both username and password")
                else:
                    success, message = self.authenticate_user(username, password)
                    if success:
                        self.create_session(username)
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
        
        # Show default admin credentials warning
        if "admin" in self.users:
            st.warning("⚠️ Default admin credentials are active. Change them immediately!")
    
    def show_user_info(self):
        """Display current user information"""
        if 'session_data' in st.session_state:
            session_data = st.session_state['session_data']
            username = session_data['username']
            role = session_data['role']
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"👤 **{username}** ({role})")
            with col2:
                if st.button("Logout"):
                    self.logout()
                    st.rerun()
    
    def create_user(self, username: str, password: str, role: UserRole, 
                   created_by: str) -> Tuple[bool, str]:
        """Create new user (admin only)"""
        if username in self.users:
            return False, "Username already exists"
        
        is_strong, message = self._is_password_strong(password)
        if not is_strong:
            return False, message
        
        salt = secrets.token_hex(32)
        password_hash = self._hash_password(password, salt)
        
        new_user = User(
            username=username,
            password_hash=password_hash,
            salt=salt,
            role=role,
            created_at=time.time(),
            is_active=True
        )
        
        self.users[username] = new_user
        self._save_users()
        
        logger.info(f"New user created: {username} by {created_by}")
        return True, "User created successfully"
    
    def change_password(self, username: str, old_password: str, 
                       new_password: str) -> Tuple[bool, str]:
        """Change user password"""
        if username not in self.users:
            return False, "User not found"
        
        user = self.users[username]
        
        # Verify old password
        if not self._verify_password(old_password, user.salt, user.password_hash):
            return False, "Current password is incorrect"
        
        # Check new password strength
        is_strong, message = self._is_password_strong(new_password)
        if not is_strong:
            return False, message
        
        # Update password
        new_salt = secrets.token_hex(32)
        new_hash = self._hash_password(new_password, new_salt)
        
        user.salt = new_salt
        user.password_hash = new_hash
        self._save_users()
        
        logger.info(f"Password changed for user: {username}")
        return True, "Password changed successfully"

# Global security manager instance
security_manager = SecurityManager()