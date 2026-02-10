"""
Composio ToolSet Manager (Multi-App)

Provides LangChain-compatible tools for authenticated users across all apps.
Uses official Composio SDK patterns.
"""

import logging
import os
from typing import Dict, List, Optional, Any

from langchain_core.tools import BaseTool

# Initialize logger first
logger = logging.getLogger("composio_tools")

# Import official Composio exceptions for better error handling
try:
    from composio import exceptions as composio_exceptions
    COMPOSIO_EXCEPTIONS_AVAILABLE = True
except ImportError:
    logger.warning("Composio exceptions module not available. Using generic exception handling.")
    COMPOSIO_EXCEPTIONS_AVAILABLE = False
    composio_exceptions = None


class ComposioToolManager:
    """
    Manages LangChain tools for authenticated Composio sessions.
    
    Only returns tools for apps where the user has active connections.
    Uses official SDK: composio.tools.get(user_id=...)
    """
    
    def __init__(self):
        self.api_key = os.getenv("COMPOSIO_API_KEY")
        if not self.api_key:
            logger.error("COMPOSIO_API_KEY not set in environment")
            raise ValueError("COMPOSIO_API_KEY required")
        
        # Initialize Composio client (official SDK pattern)
        from composio import Composio
        self._composio = Composio(api_key=self.api_key)
        logger.info("Initialized Composio client for tools")
    
    def _format_composio_error(self, error: Exception) -> str:
        """
        Format Composio SDK errors into user-friendly messages.
        
        Handles official Composio exception types when available.
        """
        if not COMPOSIO_EXCEPTIONS_AVAILABLE or not composio_exceptions:
            return str(error)
        
        # Handle specific Composio exception types
        if isinstance(error, getattr(composio_exceptions, 'ApiKeyNotProvidedError', type(None))):
            return "Composio API key not configured. Please set COMPOSIO_API_KEY in environment."
        elif isinstance(error, getattr(composio_exceptions, 'ComposioSDKTimeoutError', type(None))):
            return "Request timeout. Please try again."
        elif isinstance(error, getattr(composio_exceptions, 'NoItemsFound', type(None))):
            return "No connected accounts found for user. Please connect the app first."
        else:
            error_str = str(error)
            if "401" in error_str or "unauthorized" in error_str.lower():
                return "Authentication failed. User may need to reconnect the app."
            elif "connection" in error_str.lower() and "not found" in error_str.lower():
                return "User not connected to this app. Please connect first."
            else:
                return error_str
    
    def get_tools_for_user(
        self,
        user_id: str,
        toolkits: Optional[List[str]] = None,
        tools: Optional[List[str]] = None,
    ) -> List[BaseTool]:
        """
        Get LangChain tools for authenticated user using official SDK.
        
        Args:
            user_id: Your app's user ID (REQUIRED - SDK needs this)
            toolkits: Toolkit slugs like ["gmail", "github", "slack"]
            tools: Specific tool slugs like ["GMAIL_SEND_EMAIL"]
        
        Returns:
            List of LangChain BaseTool objects ready for agent use
        
        Note:
            SDK automatically filters tools based on user's active connections.
            If user doesn't have gmail connected, gmail tools won't be returned.
        
        Example:
            tools = manager.get_tools_for_user(
                user_id="user_123",
                toolkits=["gmail"]
            )
            # Only returns gmail tools if user has gmail connected
        """
        if not self.api_key:
            logger.error("Cannot get tools: COMPOSIO_API_KEY not configured")
            return []
        
        try:
            # Use official SDK pattern: composio.tools.get(user_id=...)
            if tools:
                # Get specific tools by slug
                tools_list = self._composio.tools.get(
                    user_id=user_id,  # ✅ REQUIRED
                    tools=tools
                )
                logger.info(f"Retrieved {len(tools_list)} specific tools for user {user_id}")
            elif toolkits:
                # Get tools by toolkit (recommended)
                tools_list = self._composio.tools.get(
                    user_id=user_id,  # ✅ REQUIRED
                    toolkits=toolkits  # Use strings, not App enum
                )
                logger.info(f"Retrieved {len(tools_list)} toolkit tools for user {user_id}")
            else:
                # Get all tools for user's connected apps
                logger.warning("No toolkits specified, getting all connected tools")
                tools_list = self._composio.tools.get(user_id=user_id)
                logger.info(f"Retrieved {len(tools_list)} tools for user {user_id}")
            
            return tools_list
            
        except Exception as e:
            error_msg = self._format_composio_error(e)
            logger.error(f"Failed to get tools for {user_id}: {error_msg}", exc_info=True)
            
            # Return empty list but log helpful troubleshooting info
            if "not connected" in error_msg.lower() or "no connected accounts" in error_msg.lower():
                logger.info(f"💡 Troubleshooting: User {user_id} needs to connect apps via /connections page")
            elif "api key" in error_msg.lower():
                logger.error("💡 Troubleshooting: Set COMPOSIO_API_KEY in environment variables")
            
            return []
    
    def execute_tool(
        self,
        user_id: str,
        tool_slug: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a tool for a specific user using official SDK.
        
        Args:
            user_id: Your app's user ID (REQUIRED - SDK uses this to find connection)
            tool_slug: Tool identifier (e.g., "GMAIL_SEND_EMAIL")
            arguments: Tool parameters
        
        Returns:
            Execution result with success status, data, and error
        
        Example:
            result = manager.execute_tool(
                user_id="user_123",
                tool_slug="GMAIL_SEND_EMAIL",
                arguments={
                    "to": "user@example.com",
                    "subject": "Hello",
                    "body": "Test email"
                }
            )
        """
        try:
            # Use official SDK: composio.tools.execute(user_id=...)
            result = self._composio.tools.execute(
                user_id=user_id,  # ✅ SDK uses this to find connection
                slug=tool_slug,
                arguments=arguments
            )
            
            success = result.get("successful", False)
            
            if not success:
                logger.warning(f"Tool execution unsuccessful for {tool_slug}: {result.get('error')}")
            
            return {
                "success": success,
                "data": result.get("data", {}),
                "error": result.get("error")
            }
            
        except Exception as e:
            error_msg = self._format_composio_error(e)
            logger.error(f"Tool execution failed for {user_id}/{tool_slug}: {error_msg}", exc_info=True)
            
            # Provide troubleshooting hints
            troubleshooting_hint = None
            if "not connected" in error_msg.lower():
                troubleshooting_hint = f"User needs to connect the app at /connections page"
            elif "401" in error_msg or "unauthorized" in error_msg.lower():
                troubleshooting_hint = "User may need to refresh or reconnect the app"
            
            return {
                "success": False,
                "data": {},
                "error": error_msg,
                "troubleshooting": troubleshooting_hint
            }
    
    def get_invoice_tools_for_user(self, user_id: str) -> List[BaseTool]:
        """
        Get invoice-specific tools for a user (Zoho Books).
        
        This is a convenience method that gets Zoho Books tools.
        Requires user to be connected to Zoho Books.
        
        Args:
            user_id: Your app's user ID
        
        Returns:
            List of Zoho Books tools (empty if not connected)
        """
        try:
            # Get only Zoho Books tools
            tools = self._composio.tools.get(
                user_id=user_id,
                toolkits=["zohobooks"]  # or "zoho_books"
            )
            
            logger.info(f"Retrieved {len(tools)} Zoho Books tools for user {user_id}")
            return tools
        
        except Exception as e:
            error_msg = self._format_composio_error(e)
            logger.error(f"Failed to get invoice tools: {error_msg}", exc_info=True)
            return []


_tool_manager = None


def get_tool_manager() -> ComposioToolManager:
    """Get or create singleton tool manager."""
    global _tool_manager
    if _tool_manager is None:
        _tool_manager = ComposioToolManager()
    return _tool_manager


def get_tools_for_user(
    user_id: str, 
    toolkits: Optional[List[str]] = None,
    tools: Optional[List[str]] = None
) -> List[BaseTool]:
    """
    Convenience function to get tools for a user.
    
    Args:
        user_id: Your app's user ID (REQUIRED)
        toolkits: Toolkit slugs like ["gmail", "github"]
        tools: Specific tool slugs like ["GMAIL_SEND_EMAIL"]
    
    Returns:
        List of LangChain BaseTool objects
    
    Example:
        from services.integrations.composio_tools import get_tools_for_user
        
        # Get all gmail tools for user
        tools = get_tools_for_user("user_123", toolkits=["gmail"])
        
        # Get specific tools
        tools = get_tools_for_user(
            "user_123", 
            tools=["GMAIL_SEND_EMAIL", "GMAIL_SEARCH_EMAILS"]
        )
    """
    manager = get_tool_manager()
    return manager.get_tools_for_user(user_id, toolkits=toolkits, tools=tools)
