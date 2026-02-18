# agents/mail_agent/__init__.py
"""
Mail Agent - Modularized package for Gmail integration.
"""

import os
import sys

# Add backend to path for imports
from pathlib import Path
backend_root = Path(__file__).parent.parent.parent.resolve()
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

try:
    from .config import AGENT_DEFINITION, COMPOSIO_API_KEY, MCP_URL, CONNECTION_ID
    from .client import GmailClient, gmail_client
    from .agent import app as mail_app
    from .agent import app  # Also export as 'app' for uvicorn
    
    # Check configuration on startup
    if not COMPOSIO_API_KEY:
        print("WARNING: COMPOSIO_API_KEY not set. Mail agent may not function properly.", file=sys.stderr)
    if not MCP_URL:
        print("WARNING: GMAIL_MCP_URL not set. Mail agent may not function properly.", file=sys.stderr)
        
except Exception as e:
    print(f"ERROR importing mail agent: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    
    # Create a minimal app that returns error
    from fastapi import FastAPI
    app = FastAPI(title="Mail Agent - ERROR")
    
    @app.get("/health")
    async def health():
        return {"status": "error", "message": f"Import error: {str(e)}"}
    
    @app.post("/execute")
    async def execute(request):
        return {"status": "error", "error": f"Agent failed to start: {str(e)}"}
