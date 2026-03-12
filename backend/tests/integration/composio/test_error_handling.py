"""
Composio Error Handling Integration Tests

Tests error handling in composio_auth.py and composio_tools.py:
- Invalid connection ID handling
- Error message formatting
- Timeout scenarios
- Various HTTP error codes (401, 404, 429, 500)

Tasks: 2.5.1, 2.5.2, 2.5.3
"""

import os
import pytest
from unittest.mock import patch, MagicMock

# Import the managers we're testing
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))
from services.integrations.composio_auth import ComposioAuthManager, get_auth_manager
from services.integrations.composio_tools import ComposioToolManager, get_tool_manager


class TestInvalidConnectionHandling:
    """
    Task 2.5.1: Test graceful handling of invalid connection ID
    
    Verifies that invalid connection IDs return user-friendly errors
    rather than technical exceptions.
    """
    
    def test_check_connection_with_invalid_id(self, test_user_id):
        """Test checking status with non-existent connection ID."""
        auth_manager = get_auth_manager()
        
        # Use a non-existent app slug
        result = auth_manager.check_connection_status(test_user_id, "nonexistent_app_xyz")
        
        # Should return graceful error, not crash
        assert result is not None
        assert isinstance(result, dict)
        assert "success" in result
        
        # If not connected, should indicate that clearly
        if not result.get("success"):
            assert "error" in result or "connected_apps" in result
    
    def test_get_connection_for_invalid_app(self, test_user_id):
        """Test retrieving connection for app user hasn't connected."""
        auth_manager = get_auth_manager()
        
        # Try to get connection for an app that doesn't exist
        connection = auth_manager.get_connection_for_agent(test_user_id, "fake_app_12345")
        
        # Should return None gracefully, not raise exception
        assert connection is None
    
    def test_disconnect_nonexistent_connection(self, test_user_id):
        """Test disconnecting an app that was never connected."""
        auth_manager = get_auth_manager()
        
        # Try to disconnect non-existent connection
        result = auth_manager.disconnect_app(test_user_id, "never_connected_app")
        
        # Should handle gracefully
        assert result is not None
        assert isinstance(result, dict)
        assert "success" in result
        
        # May succeed (no-op) or fail gracefully with message
        if not result.get("success"):
            assert "error" in result
            error_msg = result["error"].lower()
            # Should be user-friendly, not technical
            assert "exception" not in error_msg
            assert "traceback" not in error_msg
    
    def test_refresh_invalid_connection(self, test_user_id):
        """Test refreshing a connection that doesn't exist."""
        auth_manager = get_auth_manager()
        
        # Try to refresh non-existent connection
        result = auth_manager.refresh_connection(test_user_id, "invalid_app_999")
        
        # Should return error response, not crash
        assert result is not None
        assert isinstance(result, dict)
        assert "success" in result
        assert result["success"] is False
        assert "error" in result
        
        # Error should be user-friendly
        error_msg = result["error"]
        assert len(error_msg) > 0
        assert "not found" in error_msg.lower() or "connect" in error_msg.lower()
    
    def test_get_tools_for_unconnected_app(self, test_user_id):
        """Test getting tools when user hasn't connected the app."""
        tool_manager = get_tool_manager()
        
        # Try to get tools for app user hasn't connected
        tools = tool_manager.get_tools_for_user(
            user_id=test_user_id,
            toolkits=["nonexistent_toolkit_xyz"]
        )
        
        # Should return empty list or handle gracefully
        assert isinstance(tools, list)
        # Empty list is acceptable - means no tools available


class TestErrorFormatting:
    """
    Task 2.5.2: Test error formatting for user-friendly messages
    
    Verifies that _format_composio_error produces user-friendly messages
    for various error conditions.
    """
    
    def test_format_401_unauthorized_error(self):
        """Test formatting of 401 unauthorized errors."""
        auth_manager = get_auth_manager()
        
        # Create a mock error with 401 status
        error = Exception("401 Unauthorized: Invalid credentials")
        formatted = auth_manager._format_composio_error(error)
        
        # Should be user-friendly
        assert "Authentication failed" in formatted or "reconnect" in formatted.lower()
        assert "401" not in formatted  # Should not expose HTTP codes
    
    def test_format_404_not_found_error(self):
        """Test formatting of 404 not found errors."""
        auth_manager = get_auth_manager()
        
        # Create a mock error with 404 status
        error = Exception("404 Not Found: Connection not found")
        formatted = auth_manager._format_composio_error(error)
        
        # Should be user-friendly
        assert "not found" in formatted.lower() or "deleted" in formatted.lower()
        assert len(formatted) > 0
    
    def test_format_429_rate_limit_error(self):
        """Test formatting of 429 rate limit errors."""
        auth_manager = get_auth_manager()
        
        # Create a mock error with rate limit message
        error = Exception("429 Too Many Requests: Rate limit exceeded")
        formatted = auth_manager._format_composio_error(error)
        
        # Should mention rate limit and suggest retry
        assert "rate limit" in formatted.lower() or "try again later" in formatted.lower()
        assert "429" not in formatted  # Should not expose HTTP codes
    
    def test_format_500_server_error(self):
        """Test formatting of 500 internal server errors."""
        auth_manager = get_auth_manager()
        
        # Create a mock error with 500 status
        error = Exception("500 Internal Server Error")
        formatted = auth_manager._format_composio_error(error)
        
        # Should be user-friendly and suggest retry
        assert "try again" in formatted.lower() or "service error" in formatted.lower()
        assert "500" not in formatted  # Should not expose HTTP codes
    
    def test_format_timeout_error(self):
        """Test formatting of timeout errors."""
        auth_manager = get_auth_manager()
        
        # Create a mock timeout error
        error = Exception("Request timeout after 30 seconds")
        formatted = auth_manager._format_composio_error(error)
        
        # Should mention timeout and suggest retry
        assert "timeout" in formatted.lower() or "try again" in formatted.lower()
        assert len(formatted) > 0
    
    def test_format_connection_not_found_error(self):
        """Test formatting of connection not found errors."""
        auth_manager = get_auth_manager()
        
        # Create a mock connection not found error
        error = Exception("Connection not found for user")
        formatted = auth_manager._format_composio_error(error)
        
        # Should suggest connecting the app
        assert "connect" in formatted.lower() or "not found" in formatted.lower()
        assert len(formatted) > 0
    
    def test_tool_manager_error_formatting(self):
        """Test that tool manager also formats errors properly."""
        tool_manager = get_tool_manager()
        
        # Test various error types
        errors = [
            Exception("401 Unauthorized"),
            Exception("Connection not found"),
            Exception("Rate limit exceeded"),
        ]
        
        for error in errors:
            formatted = tool_manager._format_composio_error(error)
            
            # Should be user-friendly
            assert len(formatted) > 0
            assert "exception" not in formatted.lower()
            assert "traceback" not in formatted.lower()
    
    def test_error_messages_are_actionable(self):
        """Test that error messages provide actionable guidance."""
        auth_manager = get_auth_manager()
        
        # Test various error scenarios
        test_cases = [
            (Exception("401 Unauthorized"), ["reconnect", "authentication"]),
            (Exception("404 Not Found"), ["not found", "deleted"]),
            (Exception("429 Rate Limit"), ["rate limit", "try again", "later"]),
            (Exception("Connection not found"), ["connect", "not found"]),
        ]
        
        for error, expected_keywords in test_cases:
            formatted = auth_manager._format_composio_error(error).lower()
            
            # Should contain at least one actionable keyword
            assert any(keyword in formatted for keyword in expected_keywords), \
                f"Error message '{formatted}' should contain one of {expected_keywords}"


class TestTimeoutHandling:
    """
    Task 2.5.3: Test timeout handling
    
    Verifies that timeout scenarios are handled gracefully with
    appropriate error messages and retry logic.
    """
    
    @patch('services.integrations.composio_auth.Composio')
    def test_auth_flow_timeout(self, mock_composio_class, test_user_id):
        """Test timeout during OAuth flow initiation."""
        # Mock Composio client to raise timeout
        mock_client = MagicMock()
        mock_composio_class.return_value = mock_client
        
        # Simulate timeout exception
        mock_client.connected_accounts.link.side_effect = Exception("Request timeout")
        
        # Create auth manager with mocked client
        auth_manager = ComposioAuthManager()
        auth_manager._composio = mock_client
        
        # Try to start auth flow
        result = auth_manager.start_auth_flow(test_user_id, "gmail")
        
        # Should handle timeout gracefully
        assert result is not None
        assert isinstance(result, dict)
        assert "success" in result
        assert result["success"] is False
        assert "error" in result
        
        # Error message should mention timeout
        error_msg = result["error"].lower()
        assert "timeout" in error_msg or "try again" in error_msg
    
    @patch('services.integrations.composio_auth.Composio')
    def test_connection_check_timeout(self, mock_composio_class, test_user_id):
        """Test timeout during connection status check."""
        # Mock Composio client to raise timeout
        mock_client = MagicMock()
        mock_composio_class.return_value = mock_client
        
        # Simulate timeout exception
        mock_client.connected_accounts.list.side_effect = Exception("Connection timeout")
        
        # Create auth manager with mocked client
        auth_manager = ComposioAuthManager()
        auth_manager._composio = mock_client
        
        # Try to check connection status
        result = auth_manager.check_connection_status(test_user_id, "gmail")
        
        # Should handle timeout gracefully
        assert result is not None
        assert isinstance(result, dict)
        
        # Should indicate failure or timeout
        if "success" in result:
            if not result["success"]:
                assert "error" in result
    
    @patch('services.integrations.composio_tools.Composio')
    def test_tool_execution_timeout(self, mock_composio_class, test_user_id):
        """Test timeout during tool execution."""
        # Mock Composio client to raise timeout
        mock_client = MagicMock()
        mock_composio_class.return_value = mock_client
        
        # Simulate timeout exception
        mock_client.tools.execute.side_effect = Exception("Execution timeout after 30s")
        
        # Create tool manager with mocked client
        tool_manager = ComposioToolManager()
        tool_manager._composio = mock_client
        
        # Try to execute a tool
        result = tool_manager.execute_tool(
            user_id=test_user_id,
            tool_slug="GMAIL_SEND_EMAIL",
            arguments={"to": "test@example.com", "subject": "Test"}
        )
        
        # Should handle timeout gracefully
        assert result is not None
        assert isinstance(result, dict)
        assert "success" in result
        assert result["success"] is False
        assert "error" in result
        
        # Error message should be user-friendly
        error_msg = result["error"].lower()
        assert "timeout" in error_msg or "try again" in error_msg
    
    @patch('services.integrations.composio_auth.Composio')
    def test_refresh_connection_timeout(self, mock_composio_class, test_user_id):
        """Test timeout during connection refresh."""
        # Mock Composio client to raise timeout
        mock_client = MagicMock()
        mock_composio_class.return_value = mock_client
        
        # Simulate timeout exception
        mock_client.connected_accounts.list.side_effect = Exception("Timeout refreshing token")
        
        # Create auth manager with mocked client
        auth_manager = ComposioAuthManager()
        auth_manager._composio = mock_client
        
        # Try to refresh connection
        result = auth_manager.refresh_connection(test_user_id, "gmail")
        
        # Should handle timeout gracefully
        assert result is not None
        assert isinstance(result, dict)
        assert "success" in result
        assert result["success"] is False
        assert "error" in result
        
        # Error should be user-friendly
        error_msg = result["error"]
        assert len(error_msg) > 0
        assert "exception" not in error_msg.lower()
    
    def test_timeout_error_format_consistency(self):
        """Test that timeout errors are formatted consistently."""
        auth_manager = get_auth_manager()
        
        # Test various timeout error messages
        timeout_errors = [
            Exception("Request timeout"),
            Exception("Connection timeout after 30 seconds"),
            Exception("Timeout waiting for response"),
            Exception("Read timeout"),
        ]
        
        for error in timeout_errors:
            formatted = auth_manager._format_composio_error(error)
            
            # All should mention timeout or retry
            assert "timeout" in formatted.lower() or "try again" in formatted.lower()
            # Should be user-friendly
            assert len(formatted) > 0
            assert "exception" not in formatted.lower()


class TestErrorRecovery:
    """
    Additional tests for error recovery and resilience.
    
    Verifies that the system can recover from errors and continue
    operating normally.
    """
    
    def test_error_does_not_crash_manager(self, test_user_id):
        """Test that errors don't crash the auth manager."""
        auth_manager = get_auth_manager()
        
        # Try multiple operations that might fail
        operations = [
            lambda: auth_manager.check_connection_status(test_user_id, "invalid_app"),
            lambda: auth_manager.get_connection_for_agent(test_user_id, "fake_app"),
            lambda: auth_manager.disconnect_app(test_user_id, "nonexistent"),
        ]
        
        for operation in operations:
            try:
                result = operation()
                # Should return a result, not crash
                assert result is not None
            except Exception as e:
                # If it does raise, should be a handled exception
                pytest.fail(f"Operation raised unhandled exception: {e}")
    
    def test_tool_manager_error_resilience(self, test_user_id):
        """Test that tool manager handles errors without crashing."""
        tool_manager = get_tool_manager()
        
        # Try operations that might fail
        try:
            # Get tools for non-existent toolkit
            tools = tool_manager.get_tools_for_user(
                user_id=test_user_id,
                toolkits=["nonexistent_toolkit"]
            )
            assert isinstance(tools, list)
        except Exception as e:
            pytest.fail(f"Tool manager crashed on error: {e}")
    
    def test_multiple_errors_in_sequence(self, test_user_id):
        """Test handling multiple errors in sequence."""
        auth_manager = get_auth_manager()
        
        # Perform multiple operations that will fail
        for i in range(5):
            result = auth_manager.check_connection_status(
                test_user_id, 
                f"fake_app_{i}"
            )
            
            # Each should handle gracefully
            assert result is not None
            assert isinstance(result, dict)
    
    def test_error_logging_does_not_expose_secrets(self, test_user_id):
        """Test that error messages don't expose sensitive information."""
        auth_manager = get_auth_manager()
        
        # Try various operations
        result = auth_manager.check_connection_status(test_user_id, "test_app")
        
        # If there's an error, check it doesn't contain sensitive data
        if isinstance(result, dict) and "error" in result:
            error_msg = result["error"].lower()
            
            # Should not contain API keys or tokens
            assert "api_key" not in error_msg
            assert "token" not in error_msg
            assert "secret" not in error_msg
            assert "password" not in error_msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
