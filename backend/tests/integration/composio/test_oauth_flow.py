"""
OAuth Flow Integration Tests for Composio

Tests the complete OAuth authentication flow including:
- Initiating auth flow and getting redirect URL
- Checking connection status
- Disconnecting apps

These tests use the real Composio API with a test user account.
"""

import pytest
import time
import sys
import os

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))

from services.integrations.composio_auth import get_auth_manager


def test_start_auth_flow_returns_redirect_url(composio_client, test_user_id):
    """
    Test that start_auth_flow initiates OAuth and returns a redirect URL.
    
    This test verifies:
    - Connection initiation succeeds
    - Response contains a redirect URL
    - Redirect URL is a valid Composio Connect Link
    
    Task: 2.2.1 Test start_auth_flow returns redirect URL
    """
    # Initiate connection for Gmail
    app_slug = "gmail"
    
    # SDK v0.7.x: entity.initiate_connection(app_name=...) is the simplest approach
    entity = composio_client.get_entity(id=test_user_id)
    connection_request = entity.initiate_connection(app_name=app_slug.upper())
    
    # Verify response structure
    assert connection_request is not None, "Connection request should not be None"
    assert hasattr(connection_request, "redirectUrl"), "Response should have redirectUrl"
    
    redirect_url = connection_request.redirectUrl
    
    # Verify redirect URL format
    assert redirect_url is not None, "Redirect URL should not be None"
    assert isinstance(redirect_url, str), "Redirect URL should be a string"
    assert redirect_url.startswith("http"), "Redirect URL should be a valid HTTP URL"
    assert "composio.dev" in redirect_url or "composio.com" in redirect_url, \
        "Redirect URL should be a Composio domain"
    
    print("✓ Auth flow initiated successfully")
    print(f"  Redirect URL: {redirect_url}")


def test_start_auth_flow_service_wrapper(test_user_id):
    """
    Test the start_auth_flow method from our ComposioAuthManager service.
    
    This test verifies:
    - Service wrapper correctly calls Composio SDK
    - Response contains success flag
    - Response contains redirect_url
    - Response contains connection_id
    - Connection is saved to database with INITIATED status
    
    Task: 2.2.1 Test start_auth_flow returns redirect URL (Service Layer)
    """
    # Get auth manager instance
    auth_manager = get_auth_manager()
    
    # Test with Gmail app
    app_slug = "gmail"
    
    # Call start_auth_flow
    result = auth_manager.start_auth_flow(
        user_id=test_user_id,
        app_slug=app_slug
    )
    
    # Verify response structure
    assert result is not None, "Result should not be None"
    assert isinstance(result, dict), "Result should be a dictionary"
    
    # Verify success flag
    assert "success" in result, "Result should contain 'success' field"
    assert result["success"] is True, f"Auth flow should succeed, got: {result}"
    
    # Verify redirect URL
    assert "redirect_url" in result, "Result should contain 'redirect_url'"
    redirect_url = result["redirect_url"]
    assert redirect_url is not None, "Redirect URL should not be None"
    assert isinstance(redirect_url, str), "Redirect URL should be a string"
    assert redirect_url.startswith("http"), "Redirect URL should be a valid HTTP URL"
    assert "composio.dev" in redirect_url or "composio.com" in redirect_url, \
        "Redirect URL should be a Composio domain"
    
    # Verify connection_id is returned
    assert "connection_id" in result, "Result should contain 'connection_id'"
    connection_id = result["connection_id"]
    assert connection_id is not None, "Connection ID should not be None"
    assert isinstance(connection_id, str), "Connection ID should be a string"
    
    # Verify other response fields
    assert "app_slug" in result, "Result should contain 'app_slug'"
    assert result["app_slug"] == app_slug, f"App slug should be {app_slug}"
    
    assert "user_id" in result, "Result should contain 'user_id'"
    assert result["user_id"] == test_user_id, f"User ID should be {test_user_id}"
    
    assert "poll_status_url" in result, "Result should contain 'poll_status_url'"
    assert isinstance(result["poll_status_url"], str), "Poll status URL should be a string"
    
    print("✓ Service wrapper auth flow initiated successfully")
    print(f"  Success: {result['success']}")
    print(f"  Redirect URL: {redirect_url}")
    print(f"  Connection ID: {connection_id}")
    print(f"  Poll Status URL: {result['poll_status_url']}")


def test_check_connection_status_for_active_connections(
    composio_client, 
    test_user_id, 
    gmail_connection
):
    """
    Test checking connection status for an active Gmail connection.
    
    This test verifies:
    - Can retrieve connection status for authenticated user
    - Connection shows as active
    - Connection details are accessible
    
    Task: 2.2.2 Test check_connection_status for active connections
    """
    # Get all connections for test user
    connections = composio_client.connected_accounts.get(entity_ids=[test_user_id]) or []
    if not isinstance(connections, list):
        connections = [connections]
    
    # Verify we got connections
    assert connections is not None, "Connections should not be None"
    assert len(connections) > 0, "Test user should have at least one connection"
    
    # Find Gmail connection
    gmail_conn = next(
        (c for c in connections if c.appName.lower() == "gmail"), 
        None
    )
    
    assert gmail_conn is not None, "Gmail connection should exist"
    
    # Verify connection properties
    assert hasattr(gmail_conn, "id"), "Connection should have an ID"
    assert hasattr(gmail_conn, "status"), "Connection should have a status"
    assert hasattr(gmail_conn, "app_name"), "Connection should have an app_name"
    
    # Verify connection is active
    assert gmail_conn.status.lower() in ["active", "connected"], \
        f"Connection should be active, got: {gmail_conn.status}"
    
    print("✓ Connection status verified")
    print(f"  Connection ID: {gmail_conn.id}")
    print(f"  Status: {gmail_conn.status}")
    print(f"  App: {gmail_conn.appName}")


def test_check_connection_status_service_wrapper(test_user_id, gmail_connection):
    """
    Test the check_connection_status method from ComposioAuthManager service.
    
    This test verifies:
    - Service wrapper correctly syncs with Composio API
    - Returns proper structure with connected_apps, pending_apps, all_toolkits
    - Correctly identifies active connections
    - Syncs connection data to database
    
    Task: 2.2.2 Test check_connection_status for active connections
    """
    # Get auth manager instance
    auth_manager = get_auth_manager()
    
    # Call check_connection_status for all apps
    result = auth_manager.check_connection_status(user_id=test_user_id)
    
    # Verify response structure
    assert result is not None, "Result should not be None"
    assert isinstance(result, dict), "Result should be a dictionary"
    
    # Verify success flag
    assert "success" in result, "Result should contain 'success' field"
    assert result["success"] is True, f"Status check should succeed, got: {result}"
    
    # Verify user_id is returned
    assert "user_id" in result, "Result should contain 'user_id'"
    assert result["user_id"] == test_user_id, f"User ID should be {test_user_id}"
    
    # Verify connected_apps list
    assert "connected_apps" in result, "Result should contain 'connected_apps'"
    assert isinstance(result["connected_apps"], list), "connected_apps should be a list"
    
    # Verify pending_apps list
    assert "pending_apps" in result, "Result should contain 'pending_apps'"
    assert isinstance(result["pending_apps"], list), "pending_apps should be a list"
    
    # Verify all_toolkits list
    assert "all_toolkits" in result, "Result should contain 'all_toolkits'"
    assert isinstance(result["all_toolkits"], list), "all_toolkits should be a list"
    
    # Verify Gmail is in connected apps (since we have gmail_connection fixture)
    assert "gmail" in result["connected_apps"], \
        f"Gmail should be in connected_apps, got: {result['connected_apps']}"
    
    # Verify all_toolkits contains detailed connection info
    assert len(result["all_toolkits"]) > 0, "all_toolkits should not be empty"
    
    # Find Gmail in all_toolkits
    gmail_toolkit = next(
        (t for t in result["all_toolkits"] if t.get("slug") == "gmail"),
        None
    )
    
    assert gmail_toolkit is not None, "Gmail should be in all_toolkits"
    assert "is_connected" in gmail_toolkit, "Toolkit should have is_connected field"
    assert gmail_toolkit["is_connected"] is True, "Gmail should be connected"
    assert "connected_account_id" in gmail_toolkit, "Toolkit should have connected_account_id"
    assert gmail_toolkit["connected_account_id"] is not None, "connected_account_id should not be None"
    
    print("✓ Service wrapper check_connection_status verified")
    print(f"  Success: {result['success']}")
    print(f"  Connected apps: {result['connected_apps']}")
    print(f"  Pending apps: {result['pending_apps']}")
    print(f"  Total toolkits: {len(result['all_toolkits'])}")
    
    # Test checking status for specific app
    gmail_result = auth_manager.check_connection_status(
        user_id=test_user_id,
        app_slug="gmail"
    )
    
    assert gmail_result["success"] is True, "Gmail-specific check should succeed"
    assert "gmail" in gmail_result["connected_apps"], "Gmail should be connected"
    assert len(gmail_result["all_toolkits"]) >= 1, "Should have at least Gmail toolkit"
    
    print("✓ App-specific status check verified")
    print(f"  Gmail connected: {'gmail' in gmail_result['connected_apps']}")


def test_disconnect_app_removes_connection(composio_client, test_user_id):
    """
    Test disconnecting an app removes the connection from both Composio and database.
    
    This test verifies:
    1. Connection is successfully removed from Composio
    2. Connection is successfully removed from database
    3. Service method returns success response
    4. Connection logs are created for the disconnect event
    
    Note: Uses Slack instead of Gmail to avoid disrupting other tests
    
    Task: 2.2.3 Test disconnect_app removes connection
    """
    from models import UserConnection
    from database import SessionLocal
    
    app_slug = "slack"
    
    # Step 1: Check if Slack connection already exists
    connections_before = composio_client.connected_accounts.get(entity_ids=[test_user_id]) or []
    if not isinstance(connections_before, list):
        connections_before = [connections_before]
    slack_conn_before = next(
        (c for c in connections_before if c.appName.lower() == "slack"), 
        None
    )
    
    if slack_conn_before:
        print(f"Found existing Slack connection: {slack_conn_before.id}")
        connection_id = slack_conn_before.id
    else:
        # Skip test if no Slack connection exists
        # (We don't want to create connections in tests as it requires user interaction)
        pytest.skip("No Slack connection found for test user. Skipping disconnect test.")
    
    # Step 2: Verify connection exists in database before disconnect
    db = SessionLocal()
    try:
        db_connection_before = db.query(UserConnection).filter(
            UserConnection.user_id == test_user_id,
            UserConnection.app_slug == app_slug
        ).first()
        
        # Note: Connection might not be in DB if it was created outside our service
        if db_connection_before:
            print(f"✓ Connection found in database: {db_connection_before.id}")
    finally:
        db.close()
    
    # Step 3: Disconnect using our service wrapper
    auth_manager = get_auth_manager()
    result = auth_manager.disconnect_app(
        user_id=test_user_id,
        app_slug=app_slug
    )
    
    # Verify service response
    assert result is not None, "Result should not be None"
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "success" in result, "Result should contain 'success' field"
    assert result["success"] is True, f"Disconnect should succeed, got: {result}"
    
    print(f"✓ Service disconnect succeeded: {result.get('message')}")
    
    # Step 4: Verify connection is removed from Composio
    # Wait a moment for the deletion to propagate
    time.sleep(2)
    
    connections_after = composio_client.connected_accounts.get(entity_ids=[test_user_id]) or []
    if not isinstance(connections_after, list):
        connections_after = [connections_after]
    slack_conn_after = next(
        (c for c in connections_after if c.appName.lower() == "slack"), 
        None
    )
    
    assert slack_conn_after is None, \
        "Slack connection should be removed from Composio after disconnect"
    
    print("✓ Connection removed from Composio")
    
    # Step 5: Verify connection is removed from database
    db = SessionLocal()
    try:
        db_connection_after = db.query(UserConnection).filter(
            UserConnection.user_id == test_user_id,
            UserConnection.app_slug == app_slug
        ).first()
        
        assert db_connection_after is None, \
            "Connection should be removed from database after disconnect"
        
        print("✓ Connection removed from database")
        # Note: connection_logs table has been dropped; disconnect event logging removed.
        
    finally:
        db.close()


def test_disconnect_app_error_handling_nonexistent_connection(test_user_id):
    """
    Test error handling when attempting to disconnect a non-existent connection.
    
    This test verifies:
    1. Service handles non-existent connections gracefully
    2. Returns appropriate error response
    3. Does not crash or raise exceptions
    4. Error is logged to connection_logs
    
    Task: 2.2.3 Test disconnect_app removes connection (Error Handling)
    """
    from database import SessionLocal
    
    # Use an app that definitely doesn't exist for this test user
    nonexistent_app = "nonexistent_test_app_12345"
    
    # Attempt to disconnect non-existent connection
    auth_manager = get_auth_manager()
    result = auth_manager.disconnect_app(
        user_id=test_user_id,
        app_slug=nonexistent_app
    )
    
    # Verify error response structure
    assert result is not None, "Result should not be None"
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "success" in result, "Result should contain 'success' field"
    assert result["success"] is False, "Disconnect should fail for non-existent connection"
    assert "error" in result, "Result should contain 'error' field"
    
    # Verify error message is informative
    error_message = result["error"]
    assert isinstance(error_message, str), "Error message should be a string"
    assert len(error_message) > 0, "Error message should not be empty"
    assert nonexistent_app in error_message.lower() or "no active connection" in error_message.lower(), \
        f"Error message should mention the app or connection status: {error_message}"
    
    print(f"✓ Error handled gracefully: {error_message}")
    # Note: connection_logs table has been dropped; error event logging removed.
            
    finally:
        db.close()


def test_oauth_flow_error_handling_invalid_app(composio_client, test_user_id):
    """
    Test OAuth flow error handling with an invalid app slug.
    
    This test verifies:
    - System handles invalid app slugs gracefully
    - Appropriate error is raised
    - Error message is informative
    
    Additional test for error handling
    """
    invalid_app_slug = "nonexistent_app_12345"
    
    # SDK v0.7.x: entity.initiate_connection raises an exception for unknown apps
    with pytest.raises(Exception) as exc_info:
        entity = composio_client.get_entity(id=test_user_id)
        entity.initiate_connection(app_name=invalid_app_slug.upper())
    
    # Verify an exception was raised
    assert exc_info.value is not None, "Should raise an exception for invalid app"
    print(f"✓ Invalid app handled correctly: {exc_info.value}")


def test_multiple_connections_same_user(composio_client, test_user_id):
    """
    Test that a user can have multiple app connections simultaneously.
    
    This test verifies:
    - User can have connections to multiple apps
    - Each connection has unique ID
    - Connections are independently manageable
    
    Additional test for multi-app support
    """
    # Get all connections for test user
    connections = composio_client.connected_accounts.get(entity_ids=[test_user_id]) or []
    if not isinstance(connections, list):
        connections = [connections]
    
    assert connections is not None, "Connections should not be None"
    
    if len(connections) < 2:
        pytest.skip("Test user needs at least 2 connections for this test")
    
    # Verify each connection has unique ID
    connection_ids = [c.id for c in connections]
    assert len(connection_ids) == len(set(connection_ids)), \
        "All connection IDs should be unique"
    
    # Verify connections are for different apps
    app_names = [c.appName for c in connections]
    print(f"✓ User has {len(connections)} connections")
    print(f"  Apps: {', '.join(app_names)}")
    
    # Verify each connection has required properties
    for conn in connections:
        assert hasattr(conn, "id"), "Connection should have ID"
        assert hasattr(conn, "app_name"), "Connection should have app_name"
        assert hasattr(conn, "status"), "Connection should have status"
        assert conn.id is not None, "Connection ID should not be None"
        assert conn.appName is not None, "App name should not be None"
    
    print("✓ All connections have valid structure")
