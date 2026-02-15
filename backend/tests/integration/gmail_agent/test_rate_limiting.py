"""
Test Gmail Agent - Rate Limiting
Task 3.4.3: Test rate limiting works correctly
"""
import pytest
import asyncio
from fastapi.testclient import TestClient
from agents.gmail_agent.agent import app
from agents.gmail_agent.rate_limiter import user_rate_limiter

client = TestClient(app)

def test_rate_limiter_initialization():
    """Test that rate limiter is properly initialized"""
    from agents.gmail_agent.agent import app
    
    assert hasattr(app.state, "limiter")
    assert app.state.limiter is not None

def test_user_rate_limiter_allows_within_limit():
    """Test that user rate limiter allows requests within limit"""
    user_id = "test_user_rate_limit_1"
    operation = "test_operation"
    
    # Should allow first 5 requests
    for i in range(5):
        allowed = user_rate_limiter.is_allowed(user_id, operation, limit=5, window_seconds=60)
        assert allowed is True, f"Request {i+1} should be allowed"

def test_user_rate_limiter_blocks_over_limit():
    """Test that user rate limiter blocks requests over limit"""
    user_id = "test_user_rate_limit_2"
    operation = "test_operation"
    
    # Make 5 requests (limit)
    for i in range(5):
        user_rate_limiter.is_allowed(user_id, operation, limit=5, window_seconds=60)
    
    # 6th request should be blocked
    allowed = user_rate_limiter.is_allowed(user_id, operation, limit=5, window_seconds=60)
    assert allowed is False, "Request over limit should be blocked"

def test_user_rate_limiter_per_user_isolation():
    """Test that rate limits are isolated per user"""
    user1 = "test_user_rate_limit_3"
    user2 = "test_user_rate_limit_4"
    operation = "test_operation"
    
    # User 1 makes 5 requests (at limit)
    for i in range(5):
        user_rate_limiter.is_allowed(user1, operation, limit=5, window_seconds=60)
    
    # User 2 should still be allowed (different user)
    allowed = user_rate_limiter.is_allowed(user2, operation, limit=5, window_seconds=60)
    assert allowed is True, "Different user should have separate rate limit"

def test_user_rate_limiter_per_operation_isolation():
    """Test that rate limits are isolated per operation"""
    user_id = "test_user_rate_limit_5"
    operation1 = "send_email"
    operation2 = "search_email"
    
    # Make 5 requests for operation1 (at limit)
    for i in range(5):
        user_rate_limiter.is_allowed(user_id, operation1, limit=5, window_seconds=60)
    
    # operation2 should still be allowed (different operation)
    allowed = user_rate_limiter.is_allowed(user_id, operation2, limit=5, window_seconds=60)
    assert allowed is True, "Different operation should have separate rate limit"

@pytest.mark.asyncio
async def test_user_rate_limiter_window_expiry():
    """Test that rate limit window expires correctly"""
    user_id = "test_user_rate_limit_6"
    operation = "test_operation"
    
    # Make 3 requests with 1-second window
    for i in range(3):
        user_rate_limiter.is_allowed(user_id, operation, limit=3, window_seconds=1)
    
    # Should be at limit
    allowed = user_rate_limiter.is_allowed(user_id, operation, limit=3, window_seconds=1)
    assert allowed is False, "Should be at limit"
    
    # Wait for window to expire
    await asyncio.sleep(1.1)
    
    # Should be allowed again after window expires
    allowed = user_rate_limiter.is_allowed(user_id, operation, limit=3, window_seconds=1)
    assert allowed is True, "Should be allowed after window expires"

def test_rate_limit_configuration():
    """Test that rate limit configurations are properly defined"""
    from agents.gmail_agent.rate_limiter import RATE_LIMITS, get_rate_limit
    
    # Verify critical operations have rate limits
    assert "send_email" in RATE_LIMITS
    assert "send_draft" in RATE_LIMITS
    assert "create_draft" in RATE_LIMITS
    assert "search" in RATE_LIMITS
    assert "batch_operations" in RATE_LIMITS
    assert "default" in RATE_LIMITS
    
    # Verify rate limit format
    assert get_rate_limit("send_email") == "10/minute"
    assert get_rate_limit("batch_operations") == "5/minute"
    assert get_rate_limit("unknown_operation") == RATE_LIMITS["default"]

@pytest.mark.skip(reason="Requires running FastAPI server")
def test_send_email_rate_limit_endpoint():
    """Test that send email endpoint enforces rate limiting"""
    # This test would require a running server and multiple rapid requests
    # Skipped for now as it requires integration with actual server
    pass

@pytest.mark.skip(reason="Requires running FastAPI server")
def test_rate_limit_response_format():
    """Test that rate limit error response has correct format"""
    # This test would verify the 429 response format
    # Skipped for now as it requires integration with actual server
    pass

def test_user_rate_limiter_cleanup():
    """Test that old entries are cleaned up"""
    user_id = "test_user_rate_limit_7"
    operation = "test_operation"
    
    # Make some requests
    for i in range(3):
        user_rate_limiter.is_allowed(user_id, operation, limit=10, window_seconds=60)
    
    # Verify entries exist
    key = f"{user_id}:{operation}"
    assert key in user_rate_limiter.user_requests
    assert len(user_rate_limiter.user_requests[key]) == 3
    
    # Force cleanup
    user_rate_limiter._cleanup_old_entries()
    
    # Entries should still exist (not old enough)
    assert key in user_rate_limiter.user_requests

def test_check_user_rate_limit_function():
    """Test the check_user_rate_limit helper function"""
    from agents.gmail_agent.rate_limiter import check_user_rate_limit
    from fastapi import HTTPException
    
    user_id = "test_user_rate_limit_8"
    operation = "test_check_function"
    
    # Should not raise exception within limit
    try:
        for i in range(3):
            check_user_rate_limit(user_id, operation, limit=3, window_seconds=60)
    except HTTPException:
        pytest.fail("Should not raise exception within limit")
    
    # Should raise exception over limit
    with pytest.raises(HTTPException) as exc_info:
        check_user_rate_limit(user_id, operation, limit=3, window_seconds=60)
    
    assert exc_info.value.status_code == 429
    assert "Rate limit exceeded" in str(exc_info.value.detail)
