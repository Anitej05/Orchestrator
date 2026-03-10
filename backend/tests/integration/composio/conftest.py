"""
Pytest fixtures for Composio integration tests.

This module provides shared fixtures for testing Composio OAuth flow,
tool execution, and connection management.
"""

import os
import pytest
from composio import Composio


@pytest.fixture(scope="session")
def test_user_id():
    """
    Test user entity ID for Composio.
    
    Returns:
        str: Test user ID from environment or default value
    """
    return os.getenv("COMPOSIO_TEST_USER_ID", "test_user_orbimesh_ci")


@pytest.fixture(scope="session")
def composio_client():
    """
    Composio SDK client for tests.
    
    Uses session scope to reuse the same client across all tests.
    
    Returns:
        Composio: Initialized Composio client
    """
    api_key = os.getenv("COMPOSIO_API_KEY")
    if not api_key:
        pytest.skip("COMPOSIO_API_KEY not set in environment")
    return Composio(api_key=api_key)


@pytest.fixture
def gmail_connection(test_user_id, composio_client):
    """
    Pre-connected Gmail account for tests.
    
    This fixture checks if the test user has an active Gmail connection.
    If not connected, the test is skipped gracefully.
    
    Args:
        test_user_id: Test user entity ID
        composio_client: Composio SDK client
    
    Returns:
        ConnectedAccount: Gmail connection object
    
    Raises:
        pytest.skip: If Gmail is not connected for test user
    """
    connections = composio_client.connected_accounts.get(entity_ids=[test_user_id]) or []
    if not isinstance(connections, list):
        connections = [connections]
    gmail_conn = next((c for c in connections if c.appName.lower() == "gmail"), None)
    if not gmail_conn:
        pytest.skip("Gmail not connected for test user")
    return gmail_conn
