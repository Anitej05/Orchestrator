# agents/gmail_agent/rate_limiter.py
"""
Rate limiting for Gmail Agent to prevent API quota exhaustion.

Gmail API Quotas (per user per day):
- Send email: 100 per day (free tier)
- Read operations: Very high limit (1B quota units/day)
- Batch operations: Should be throttled to prevent abuse
"""
import logging
from functools import wraps
from typing import Callable
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request, HTTPException

logger = logging.getLogger("gmail_agent")

# Create limiter instance
limiter = Limiter(key_func=get_remote_address)

# Rate limit configurations
RATE_LIMITS = {
    "send_email": "10/minute",  # 10 emails per minute per IP
    "send_draft": "10/minute",  # 10 draft sends per minute per IP
    "create_draft": "20/minute",  # 20 draft creations per minute per IP
    "search": "30/minute",  # 30 searches per minute per IP
    "batch_operations": "5/minute",  # 5 batch operations per minute per IP (summarize, extract actions)
    "default": "60/minute"  # Default rate limit for other operations
}

def get_rate_limit(operation: str) -> str:
    """Get rate limit for a specific operation"""
    return RATE_LIMITS.get(operation, RATE_LIMITS["default"])

def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Custom rate limit error handler"""
    logger.warning(f"Rate limit exceeded for {request.url.path} from {get_remote_address(request)}")
    
    raise HTTPException(
        status_code=429,
        detail={
            "error": "Rate limit exceeded",
            "message": "Too many requests. Please try again later.",
            "retry_after": exc.detail.split("Retry after ")[1] if "Retry after" in exc.detail else "60 seconds"
        }
    )

# Decorator for rate limiting specific operations
def rate_limited(operation: str):
    """
    Decorator to apply rate limiting to specific operations.
    
    Usage:
        @rate_limited("send_email")
        async def send_email(...):
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Apply rate limiting logic here if needed
            # For now, we'll use slowapi's limiter directly in the routes
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# User-based rate limiting (alternative to IP-based)
class UserRateLimiter:
    """
    User-based rate limiter for more granular control.
    Tracks rate limits per user_id instead of IP address.
    """
    
    def __init__(self):
        from collections import defaultdict
        from datetime import datetime, timedelta
        
        self.user_requests = defaultdict(list)
        self.cleanup_interval = timedelta(minutes=5)
        self.last_cleanup = datetime.now()
    
    def is_allowed(self, user_id: str, operation: str, limit: int, window_seconds: int) -> bool:
        """
        Check if user is allowed to perform operation.
        
        Args:
            user_id: User identifier
            operation: Operation name (e.g., "send_email")
            limit: Maximum number of requests allowed
            window_seconds: Time window in seconds
        
        Returns:
            True if allowed, False if rate limit exceeded
        """
        from datetime import datetime, timedelta
        
        now = datetime.now()
        key = f"{user_id}:{operation}"
        
        # Cleanup old entries periodically
        if now - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_entries()
        
        # Get recent requests for this user and operation
        recent_requests = [
            req_time for req_time in self.user_requests[key]
            if now - req_time < timedelta(seconds=window_seconds)
        ]
        
        # Update the list with only recent requests
        self.user_requests[key] = recent_requests
        
        # Check if limit exceeded
        if len(recent_requests) >= limit:
            logger.warning(f"Rate limit exceeded for user {user_id} on operation {operation}")
            return False
        
        # Add current request
        self.user_requests[key].append(now)
        return True
    
    def _cleanup_old_entries(self):
        """Remove old entries to prevent memory bloat"""
        from datetime import datetime, timedelta
        
        now = datetime.now()
        cutoff = now - timedelta(hours=1)
        
        for key in list(self.user_requests.keys()):
            self.user_requests[key] = [
                req_time for req_time in self.user_requests[key]
                if req_time > cutoff
            ]
            
            # Remove empty entries
            if not self.user_requests[key]:
                del self.user_requests[key]
        
        self.last_cleanup = now

# Global user rate limiter instance
user_rate_limiter = UserRateLimiter()

def check_user_rate_limit(user_id: str, operation: str, limit: int = 10, window_seconds: int = 60):
    """
    Check user rate limit and raise exception if exceeded.
    
    Args:
        user_id: User identifier
        operation: Operation name
        limit: Maximum requests allowed
        window_seconds: Time window in seconds
    
    Raises:
        HTTPException: If rate limit exceeded
    """
    if not user_rate_limiter.is_allowed(user_id, operation, limit, window_seconds):
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "message": f"Too many {operation} requests. Please try again later.",
                "retry_after": f"{window_seconds} seconds"
            }
        )
