# agents/mail_agent/__init__.py
"""
Mail Agent - Modularized package for Gmail integration.
"""

from .config import AGENT_DEFINITION, COMPOSIO_API_KEY, MCP_URL, CONNECTION_ID
from .client import GmailClient, gmail_client
from .agent import app as mail_app
