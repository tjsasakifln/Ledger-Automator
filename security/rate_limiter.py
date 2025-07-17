# security/rate_limiter.py
"""
Rate limiting and DDoS protection for Ledger Automator
Implements token bucket algorithm with Redis backend for distributed rate limiting
"""

import time
import redis
import logging
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from datetime import datetime, timedelta
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RateLimitType(Enum):
    """Rate limit types for different operations"""
    LOGIN = "login"
    API_CALL = "api_call"
    FILE_UPLOAD = "file_upload"
    PREDICTION = "prediction"
    GENERAL = "general"

@dataclass
class RateLimitRule:
    """Rate limit rule configuration"""
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    burst_limit: int
    cooldown_minutes: int = 5

class RateLimiter:
    """Redis-based distributed rate limiter"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        
        # Default rate limit rules
        self.rules = {
            RateLimitType.LOGIN: RateLimitRule(
                requests_per_minute=5,
                requests_per_hour=20,
                requests_per_day=100,
                burst_limit=3,
                cooldown_minutes=15
            ),
            RateLimitType.API_CALL: RateLimitRule(
                requests_per_minute=60,
                requests_per_hour=1000,
                requests_per_day=10000,
                burst_limit=10,
                cooldown_minutes=1
            ),
            RateLimitType.FILE_UPLOAD: RateLimitRule(
                requests_per_minute=2,
                requests_per_hour=10,
                requests_per_day=50,
                burst_limit=1,
                cooldown_minutes=5
            ),
            RateLimitType.PREDICTION: RateLimitRule(
                requests_per_minute=30,
                requests_per_hour=500,
                requests_per_day=5000,
                burst_limit=5,
                cooldown_minutes=2
            ),
            RateLimitType.GENERAL: RateLimitRule(
                requests_per_minute=100,
                requests_per_hour=2000,
                requests_per_day=20000,
                burst_limit=20,
                cooldown_minutes=1
            )
        }
    
    def _get_client_id(self, request_context: Dict[str, Any]) -> str:
        """Generate client ID for rate limiting"""
        # Try to get IP address
        ip_address = "unknown"
        if 'ip' in request_context:
            ip_address = request_context['ip']
        elif hasattr(st, 'context') and hasattr(st.context, 'headers'):
            # Try to get from Streamlit context
            headers = st.context.headers
            ip_address = headers.get('X-Forwarded-For', headers.get('X-Real-IP', 'unknown'))
        
        # Include user ID if available
        user_id = request_context.get('user_id', 'anonymous')
        
        # Create composite client ID
        client_data = f"{ip_address}:{user_id}"
        return hashlib.sha256(client_data.encode()).hexdigest()[:16]
    
    def _get_redis_keys(self, client_id: str, limit_type: RateLimitType) -> Dict[str, str]:
        """Generate Redis keys for rate limiting"""
        base_key = f"rate_limit:{limit_type.value}:{client_id}"
        return {
            'minute': f"{base_key}:minute",
            'hour': f"{base_key}:hour",
            'day': f"{base_key}:day",
            'burst': f"{base_key}:burst",
            'cooldown': f"{base_key}:cooldown"
        }
    
    def is_allowed(self, limit_type: RateLimitType, 
                   request_context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is allowed based on rate limits"""
        try:
            client_id = self._get_client_id(request_context)
            rule = self.rules[limit_type]
            keys = self._get_redis_keys(client_id, limit_type)
            
            current_time = time.time()
            current_minute = int(current_time // 60)
            current_hour = int(current_time // 3600)
            current_day = int(current_time // 86400)
            
            # Check cooldown period
            cooldown_key = keys['cooldown']
            cooldown_end = self.redis_client.get(cooldown_key)
            if cooldown_end and float(cooldown_end) > current_time:
                return False, {
                    'error': 'Rate limit exceeded - cooldown period active',
                    'cooldown_ends': datetime.fromtimestamp(float(cooldown_end)).isoformat(),
                    'retry_after': int(float(cooldown_end) - current_time)
                }
            
            # Check rate limits
            with self.redis_client.pipeline() as pipe:
                pipe.multi()
                
                # Get current counts
                pipe.get(keys['minute'])
                pipe.get(keys['hour'])
                pipe.get(keys['day'])
                pipe.get(keys['burst'])
                
                results = pipe.execute()
                
                minute_count = int(results[0] or 0)
                hour_count = int(results[1] or 0)
                day_count = int(results[2] or 0)
                burst_count = int(results[3] or 0)
                
                # Check limits
                if minute_count >= rule.requests_per_minute:
                    self._apply_cooldown(client_id, limit_type, current_time)
                    return False, {
                        'error': 'Rate limit exceeded - too many requests per minute',
                        'limit': rule.requests_per_minute,
                        'current': minute_count,
                        'reset_time': (current_minute + 1) * 60
                    }
                
                if hour_count >= rule.requests_per_hour:
                    self._apply_cooldown(client_id, limit_type, current_time)
                    return False, {
                        'error': 'Rate limit exceeded - too many requests per hour',
                        'limit': rule.requests_per_hour,
                        'current': hour_count,
                        'reset_time': (current_hour + 1) * 3600
                    }
                
                if day_count >= rule.requests_per_day:
                    self._apply_cooldown(client_id, limit_type, current_time)
                    return False, {
                        'error': 'Rate limit exceeded - too many requests per day',
                        'limit': rule.requests_per_day,
                        'current': day_count,
                        'reset_time': (current_day + 1) * 86400
                    }
                
                if burst_count >= rule.burst_limit:
                    return False, {
                        'error': 'Rate limit exceeded - burst limit reached',
                        'limit': rule.burst_limit,
                        'current': burst_count,
                        'retry_after': 60  # Wait 1 minute for burst recovery
                    }
                
                # Increment counters
                pipe.multi()
                pipe.incr(keys['minute'])
                pipe.expire(keys['minute'], 60)
                pipe.incr(keys['hour'])
                pipe.expire(keys['hour'], 3600)
                pipe.incr(keys['day'])
                pipe.expire(keys['day'], 86400)
                pipe.incr(keys['burst'])
                pipe.expire(keys['burst'], 60)  # Burst counter resets every minute
                
                pipe.execute()
                
                # Log rate limit check
                logger.info(f"Rate limit check: {limit_type.value} for {client_id[:8]}... - allowed")
                
                return True, {
                    'allowed': True,
                    'remaining': {
                        'minute': rule.requests_per_minute - minute_count - 1,
                        'hour': rule.requests_per_hour - hour_count - 1,
                        'day': rule.requests_per_day - day_count - 1,
                        'burst': rule.burst_limit - burst_count - 1
                    }
                }
        
        except Exception as e:
            logger.error(f"Rate limiting error: {str(e)}")
            # Fail open - allow request if rate limiting fails
            return True, {'error': 'Rate limiting temporarily unavailable'}
    
    def _apply_cooldown(self, client_id: str, limit_type: RateLimitType, current_time: float):
        """Apply cooldown period for exceeded rate limits"""
        rule = self.rules[limit_type]
        keys = self._get_redis_keys(client_id, limit_type)
        
        cooldown_end = current_time + (rule.cooldown_minutes * 60)
        self.redis_client.setex(keys['cooldown'], 
                               rule.cooldown_minutes * 60, 
                               str(cooldown_end))
        
        logger.warning(f"Rate limit cooldown applied: {limit_type.value} for {client_id[:8]}... - {rule.cooldown_minutes} minutes")
    
    def reset_limits(self, client_id: str, limit_type: RateLimitType):
        """Reset rate limits for a client (admin function)"""
        keys = self._get_redis_keys(client_id, limit_type)
        
        with self.redis_client.pipeline() as pipe:
            pipe.delete(keys['minute'])
            pipe.delete(keys['hour'])
            pipe.delete(keys['day'])
            pipe.delete(keys['burst'])
            pipe.delete(keys['cooldown'])
            pipe.execute()
        
        logger.info(f"Rate limits reset for {client_id[:8]}... - {limit_type.value}")
    
    def get_client_stats(self, client_id: str, limit_type: RateLimitType) -> Dict[str, Any]:
        """Get current rate limit statistics for a client"""
        keys = self._get_redis_keys(client_id, limit_type)
        rule = self.rules[limit_type]
        
        with self.redis_client.pipeline() as pipe:
            pipe.get(keys['minute'])
            pipe.get(keys['hour'])
            pipe.get(keys['day'])
            pipe.get(keys['burst'])
            pipe.get(keys['cooldown'])
            
            results = pipe.execute()
            
            minute_count = int(results[0] or 0)
            hour_count = int(results[1] or 0)
            day_count = int(results[2] or 0)
            burst_count = int(results[3] or 0)
            cooldown_end = results[4]
            
            stats = {
                'client_id': client_id,
                'limit_type': limit_type.value,
                'current_usage': {
                    'minute': minute_count,
                    'hour': hour_count,
                    'day': day_count,
                    'burst': burst_count
                },
                'limits': {
                    'minute': rule.requests_per_minute,
                    'hour': rule.requests_per_hour,
                    'day': rule.requests_per_day,
                    'burst': rule.burst_limit
                },
                'remaining': {
                    'minute': max(0, rule.requests_per_minute - minute_count),
                    'hour': max(0, rule.requests_per_hour - hour_count),
                    'day': max(0, rule.requests_per_day - day_count),
                    'burst': max(0, rule.burst_limit - burst_count)
                },
                'cooldown_active': cooldown_end is not None,
                'cooldown_ends': cooldown_end
            }
            
            return stats

def rate_limit_decorator(limit_type: RateLimitType):
    """Decorator for rate limiting functions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get request context
            request_context = {
                'user_id': getattr(st.session_state, 'user_id', 'anonymous'),
                'function': func.__name__
            }
            
            # Check rate limit
            allowed, info = rate_limiter.is_allowed(limit_type, request_context)
            
            if not allowed:
                # Display rate limit error to user
                if 'retry_after' in info:
                    st.error(f"⏰ Rate limit exceeded. Please wait {info['retry_after']} seconds before trying again.")
                else:
                    st.error(f"⏰ Rate limit exceeded: {info.get('error', 'Too many requests')}")
                
                # Show rate limit info
                with st.expander("ℹ️ Rate Limit Information"):
                    st.json(info)
                
                return None
            
            # Execute function if allowed
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

def check_rate_limit_streamlit(limit_type: RateLimitType, 
                              show_info: bool = False) -> bool:
    """Check rate limit in Streamlit context"""
    request_context = {
        'user_id': getattr(st.session_state, 'user_id', 'anonymous'),
    }
    
    allowed, info = rate_limiter.is_allowed(limit_type, request_context)
    
    if not allowed:
        if 'retry_after' in info:
            st.error(f"⏰ Rate limit exceeded. Please wait {info['retry_after']} seconds before trying again.")
        else:
            st.error(f"⏰ Rate limit exceeded: {info.get('error', 'Too many requests')}")
        
        if show_info:
            with st.expander("ℹ️ Rate Limit Information"):
                st.json(info)
    
    elif show_info and 'remaining' in info:
        st.info(f"✅ Request allowed. Remaining: {info['remaining']}")
    
    return allowed

# Global rate limiter instance
rate_limiter = RateLimiter()