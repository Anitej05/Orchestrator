"""
Composio Integration Module

Handles OAuth authentication and tool management for multiple external apps.
"""

from .composio_auth import get_auth_manager
from .composio_tools import get_tool_manager

__all__ = ["get_auth_manager", "get_tool_manager"]
