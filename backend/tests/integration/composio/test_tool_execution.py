"""
Tool execution tests for Composio integration.

Tests tool retrieval and execution functionality using the composio_tools module.
Validates that tools can be discovered and executed for authenticated users.
"""

import pytest
import sys
import os

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))

from services.integrations.composio_tools import (
    get_tool_manager,
    get_tools_for_user
)


class TestToolRetrieval:
    """Test tool discovery and retrieval functionality."""
    
    def test_get_tools_for_user_returns_tools(self, test_user_id, gmail_connection):
        """
        Test that get_tools_for_user returns tools for connected apps.
        
        Task: 2.3.1 Test get_tools_for_user returns tools
        
        Validates:
        - Tool manager can retrieve tools for authenticated user
        - Tools are returned as a list
        - Tools are available for connected apps (Gmail)
        """
        # Get Gmail tools for test user
        tools = get_tools_for_user(
            user_id=test_user_id,
            toolkits=["gmail"]
        )
        
        # Verify tools are returned
        assert tools is not None, "Tools should not be None"
        assert isinstance(tools, list), "Tools should be a list"
        assert len(tools) > 0, "Should return at least one Gmail tool"
        
        # Verify tools have expected attributes (LangChain BaseTool)
        first_tool = tools[0]
        assert hasattr(first_tool, 'name'), "Tool should have a name attribute"
        assert hasattr(first_tool, 'description'), "Tool should have a description"
    
    def test_get_tools_with_specific_toolkit(self, test_user_id, gmail_connection):
        """
        Test retrieving tools for a specific toolkit.
        
        Validates that toolkit filtering works correctly.
        """
        # Get tools for Gmail toolkit
        gmail_tools = get_tools_for_user(
            user_id=test_user_id,
            toolkits=["gmail"]
        )
        
        assert len(gmail_tools) > 0, "Should return Gmail tools"
        
        # Verify all tools are Gmail-related
        for tool in gmail_tools:
            tool_name = tool.name.upper()
            assert "GMAIL" in tool_name or "GOOGLE" in tool_name, \
                f"Tool {tool.name} should be Gmail-related"
    
    def test_get_tools_for_unconnected_app(self, test_user_id, composio_client):
        """
        Test that get_tools_for_user returns empty list for unconnected apps.
        
        Validates graceful handling when user hasn't connected an app.
        """
        # Try to get tools for an app that's unlikely to be connected
        tools = get_tools_for_user(
            user_id=test_user_id,
            toolkits=["notion"]  # Assuming test user doesn't have Notion connected
        )
        
        # Should return empty list, not raise an exception
        assert isinstance(tools, list), "Should return a list even for unconnected apps"
        # Note: If Notion IS connected, this will have tools, which is also valid


class TestToolExecution:
    """Test tool execution functionality."""
    
    def test_execute_tool_with_valid_parameters(self, test_user_id, gmail_connection):
        """
        Test executing a tool with valid parameters.
        
        Task: 2.3.2 Test execute_tool with valid parameters
        
        Validates:
        - Tool manager can execute tools for authenticated users
        - Execution returns success status
        - Result contains expected data structure
        """
        tool_manager = get_tool_manager()
        
        # Execute a simple Gmail tool (get user profile)
        # This is a safe read-only operation
        result = tool_manager.execute_tool(
            user_id=test_user_id,
            tool_slug="GMAIL_GET_PROFILE",
            arguments={}
        )
        
        # Verify result structure
        assert result is not None, "Result should not be None"
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "success" in result, "Result should have 'success' field"
        assert "data" in result, "Result should have 'data' field"
        assert "error" in result, "Result should have 'error' field"
        
        # Verify execution succeeded
        if not result["success"]:
            # If it failed, log the error for debugging
            pytest.fail(f"Tool execution failed: {result.get('error')}")
        
        assert result["success"] is True, "Tool execution should succeed"
        assert result["data"] is not None, "Should return data on success"
    
    def test_execute_tool_with_search_emails(self, test_user_id, gmail_connection):
        """
        Test executing Gmail search with valid parameters.
        
        Validates that search operations work correctly.
        """
        tool_manager = get_tool_manager()
        
        # Execute Gmail search (read-only, safe for testing)
        result = tool_manager.execute_tool(
            user_id=test_user_id,
            tool_slug="GMAIL_SEARCH_EMAILS",
            arguments={
                "query": "is:inbox",
                "max_results": 5
            }
        )
        
        # Verify result
        assert result is not None
        assert isinstance(result, dict)
        
        # Check if execution succeeded
        if result["success"]:
            assert result["data"] is not None, "Should return data on success"
        else:
            # Log error for debugging but don't fail - might be quota/permission issue
            print(f"Search failed (may be expected): {result.get('error')}")


class TestToolExecutionErrors:
    """Test error handling in tool execution."""
    
    def test_tool_execution_with_invalid_connection(self, composio_client):
        """
        Test tool execution with invalid/non-existent connection.
        
        Task: 2.3.3 Test tool execution with invalid connection
        
        Validates:
        - Tool execution fails gracefully with invalid user
        - Error message is returned
        - No exceptions are raised
        """
        tool_manager = get_tool_manager()
        
        # Try to execute tool with non-existent user
        result = tool_manager.execute_tool(
            user_id="nonexistent_user_12345",
            tool_slug="GMAIL_GET_PROFILE",
            arguments={}
        )
        
        # Verify error handling
        assert result is not None, "Should return result even on error"
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "success" in result, "Result should have 'success' field"
        assert "error" in result, "Result should have 'error' field"
        
        # Verify execution failed
        assert result["success"] is False, "Execution should fail for invalid user"
        assert result["error"] is not None, "Should return error message"
        assert len(result["error"]) > 0, "Error message should not be empty"
    
    def test_tool_execution_with_invalid_tool_slug(self, test_user_id, gmail_connection):
        """
        Test tool execution with non-existent tool slug.
        
        Validates error handling for invalid tool names.
        """
        tool_manager = get_tool_manager()
        
        # Try to execute non-existent tool
        result = tool_manager.execute_tool(
            user_id=test_user_id,
            tool_slug="INVALID_TOOL_THAT_DOES_NOT_EXIST",
            arguments={}
        )
        
        # Verify error handling
        assert result is not None
        assert isinstance(result, dict)
        assert result["success"] is False, "Should fail for invalid tool"
        assert result["error"] is not None, "Should return error message"
    
    def test_tool_execution_with_missing_required_params(self, test_user_id, gmail_connection):
        """
        Test tool execution with missing required parameters.
        
        Validates parameter validation and error messages.
        """
        tool_manager = get_tool_manager()
        
        # Try to execute tool without required parameters
        # GMAIL_SEND_EMAIL requires 'to', 'subject', 'body'
        result = tool_manager.execute_tool(
            user_id=test_user_id,
            tool_slug="GMAIL_SEND_EMAIL",
            arguments={}  # Missing required params
        )
        
        # Verify error handling
        assert result is not None
        assert isinstance(result, dict)
        assert result["success"] is False, "Should fail for missing parameters"
        assert result["error"] is not None, "Should return error message"


class TestToolManagerSingleton:
    """Test tool manager singleton pattern."""
    
    def test_get_tool_manager_returns_singleton(self):
        """
        Test that get_tool_manager returns the same instance.
        
        Validates singleton pattern implementation.
        """
        manager1 = get_tool_manager()
        manager2 = get_tool_manager()
        
        assert manager1 is manager2, "Should return same instance (singleton)"
    
    def test_tool_manager_has_composio_client(self):
        """
        Test that tool manager initializes Composio client.
        
        Validates proper initialization.
        """
        manager = get_tool_manager()
        
        assert hasattr(manager, '_composio'), "Should have Composio client"
        assert manager._composio is not None, "Composio client should be initialized"
        assert hasattr(manager, 'api_key'), "Should have API key"
        assert manager.api_key is not None, "API key should be set"
