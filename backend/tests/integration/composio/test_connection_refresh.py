"""
Connection Refresh Integration Tests for Composio

Tests connection refresh and token management:
- Refreshing active connections via ComposioAuthManager
- Verifying database timestamp updates
- Handling refresh failures with invalid connection IDs

These tests use the real Composio API with a test user account
and verify the ComposioAuthManager.refresh_connection() method.
"""

import pytest
import time
from database import SessionLocal
from models import UserConnection
from services.integrations.composio_auth import get_auth_manager


@pytest.fixture
def auth_manager():
    """Get ComposioAuthManager instance for testing."""
    return get_auth_manager()


def test_refresh_connection_updates_timestamp(auth_manager, test_user_id, gmail_connection):
    """
    Test that refresh_connection updates the auth_timestamp in database.
    
    This test verifies:
    - refresh_connection() successfully refreshes an active connection
    - auth_timestamp in database is updated after refresh
    - Connection remains active after refresh
    
    Task: 2.4.1 Test refresh_connection updates timestamp
    
    Validates: Requirements AC-2.4 (Tests verify connection refresh mechanism works)
    """
    # First, ensure connection exists in database by syncing
    sync_result = auth_manager.check_connection_status(test_user_id, "gmail")
    assert sync_result["success"], "Should sync connection to database"
    
    # Get initial timestamp from database
    db = SessionLocal()
    try:
        initial_conn = db.query(UserConnection).filter(
            UserConnection.user_id == test_user_id,
            UserConnection.app_slug == "gmail"
        ).first()
        
        assert initial_conn is not None, "Connection should exist in database"
        initial_timestamp = initial_conn.auth_timestamp
        
        print("Initial connection:")
        print(f"  User: {test_user_id}")
        print("  App: gmail")
        print(f"  Status: {initial_conn.status}")
        print(f"  Auth timestamp: {initial_timestamp}")
        
    finally:
        db.close()
    
    # Wait a moment to ensure timestamp will be different
    time.sleep(2)
    
    # Refresh the connection using ComposioAuthManager
    result = auth_manager.refresh_connection(test_user_id, "gmail")
    
    # Verify refresh succeeded
    assert result["success"], f"Refresh should succeed: {result.get('error')}"
    assert "refreshed_at" in result, "Result should include refresh timestamp"
    
    print("✓ Refresh succeeded")
    print(f"  Refreshed at: {result['refreshed_at']}")
    
    # Get updated timestamp from database
    db = SessionLocal()
    try:
        updated_conn = db.query(UserConnection).filter(
            UserConnection.user_id == test_user_id,
            UserConnection.app_slug == "gmail"
        ).first()
        
        assert updated_conn is not None, "Connection should still exist in database"
        updated_timestamp = updated_conn.auth_timestamp
        
        print("Updated connection:")
        print(f"  Status: {updated_conn.status}")
        print(f"  Auth timestamp: {updated_timestamp}")
        
        # Verify timestamp was updated
        assert updated_timestamp > initial_timestamp, \
            f"auth_timestamp should be updated: {initial_timestamp} -> {updated_timestamp}"
        
        # Verify connection is still active
        assert updated_conn.status in ["active", "stale"], \
            f"Connection should remain active, got: {updated_conn.status}"
        
        print("✓ Database timestamp updated successfully")
        print(f"  Time difference: {(updated_timestamp - initial_timestamp).total_seconds():.2f} seconds")
        
    finally:
        db.close()


def test_refresh_with_invalid_connection_id(auth_manager):
    """
    Test refresh_connection behavior with invalid connection ID.
    
    This test verifies:
    - System handles invalid connection IDs gracefully
    - Returns error response (not exception)
    - Error message is informative
    
    Task: 2.4.2 Test refresh with invalid connection ID
    
    Validates: Requirements AC-2.5 (Tests handle connection failures gracefully)
    """
    # Use a non-existent user and app combination
    invalid_user_id = "nonexistent_user_12345"
    invalid_app_slug = "gmail"
    
    print(f"Testing refresh with invalid user: {invalid_user_id}")
    
    # Attempt to refresh connection that doesn't exist
    result = auth_manager.refresh_connection(invalid_user_id, invalid_app_slug)
    
    # Verify it returns an error (not raises exception)
    assert result is not None, "Should return a result dict"
    assert "success" in result, "Result should have 'success' field"
    assert result["success"] is False, "Should indicate failure"
    assert "error" in result, "Result should include error message"
    
    error_message = result["error"].lower()
    
    # Verify error message is informative
    assert any(keyword in error_message for keyword in [
        "not found", "no connection", "reconnect"
    ]), f"Error message should be informative: {result['error']}"
    
    print("✓ Invalid connection handled gracefully")
    print(f"  Success: {result['success']}")
    print(f"  Error: {result['error']}")
    
    # Test with invalid connection_id directly
    invalid_connection_id = "ca_nonexistent_12345"
    
    print(f"\nTesting refresh with invalid connection ID: {invalid_connection_id}")
    
    # Create a test user but provide invalid connection_id
    result2 = auth_manager.refresh_connection(
        "test_user", 
        "gmail", 
        connection_id=invalid_connection_id
    )
    
    # Should also fail gracefully
    assert result2["success"] is False, "Should indicate failure for invalid connection_id"
    assert "error" in result2, "Should include error message"
    
    print("✓ Invalid connection ID handled gracefully")
    print(f"  Success: {result2['success']}")
    print(f"  Error: {result2['error']}")

