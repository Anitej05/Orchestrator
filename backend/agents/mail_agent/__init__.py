# agents/mail_agent/__init__.py
"""
Mail Agent - Modularized package for Gmail integration.
Supports both legacy FastAPI app and new BaseAgent architecture.
"""

import os
import sys

# Add backend to path for imports
from pathlib import Path
backend_root = Path(__file__).parent.parent.parent.resolve()
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

# Legacy imports (maintain backwards compatibility)
try:
    from .config import AGENT_DEFINITION, COMPOSIO_API_KEY, MCP_URL, CONNECTION_ID
    from .client import GmailClient, gmail_client
    from .agent import app as legacy_app
    
    # Check configuration on startup
    if not COMPOSIO_API_KEY:
        print("WARNING: COMPOSIO_API_KEY not set. Mail agent may not function properly.", file=sys.stderr)
    if not MCP_URL:
        print("WARNING: GMAIL_MCP_URL not set. Mail agent may not function properly.", file=sys.stderr)
        
except Exception as e:
    print(f"WARNING: Legacy mail agent import failed: {e}", file=sys.stderr)
    legacy_app = None

# New BaseAgent implementation
try:
    from .base_agent_impl import MailAgent
    from backend.agents.base.server import create_agent_server
    
    # Create server instance for the new architecture
    _server = create_agent_server(
        agent_class=MailAgent,
        agent_id="mail_agent",
        agent_name="Mail Agent"
    )
    app = _server.app  # Export FastAPI app from BaseAgent server
    
except Exception as e:
    print(f"ERROR: BaseAgent mail agent failed to initialize: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    
    # Fallback to legacy or error app
    if legacy_app:
        app = legacy_app
    else:
        from fastapi import FastAPI
        app = FastAPI(title="Mail Agent - ERROR")
        
        @app.get("/health")
        async def health():
            return {"status": "error", "message": f"Import error: {str(e)}"}
        
        @app.post("/execute")
        async def execute(request):
            return {"status": "error", "error": f"Agent failed to start: {str(e)}"}

__all__ = [
    'MailAgent',           # New BaseAgent class
    'app',                 # FastAPI app (BaseAgent server)
    'legacy_app',          # Legacy FastAPI app (if available)
    'GmailClient',
    'gmail_client',
    'AGENT_DEFINITION',
]
