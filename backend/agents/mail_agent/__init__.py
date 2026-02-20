# agents/mail_agent/__init__.py
"""
Mail Agent - Modularized package for Gmail integration.
Supports both legacy FastAPI app and new BaseAgent architecture.
"""

from .config import AGENT_DEFINITION, COMPOSIO_API_KEY, MCP_URL, CONNECTION_ID
from .client import GmailClient, gmail_client
from .agent import app as mail_app

def run_agent() -> None:
    import uvicorn
    uvicorn.run(mail_app, host="0.0.0.0", port=8040)


if __name__ == "__main__":
    run_agent()
