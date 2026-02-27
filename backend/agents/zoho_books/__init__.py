"""
Zoho Books Agent

Zoho Books accounting integration agent.
Supports both legacy and BaseAgent architectures.
"""

# CRITICAL: Load .env BEFORE any imports that trigger InferenceService/KeyManager singletons.
from pathlib import Path as _Path
from dotenv import load_dotenv as _load_dotenv
_load_dotenv(_Path(__file__).parent.parent.parent / ".env")
# Legacy imports
try:
    from .zoho_books_agent import ZohoBooksAgent as LegacyZohoBooksAgent
    legacy_available = True
except ImportError:
    legacy_available = False

# New BaseAgent implementation
try:
    from .base_agent_impl import ZohoBooksAgent
    from backend.agents.base.server import create_agent_server
    
    # Create server instance
    _server = create_agent_server(
        agent_class=ZohoBooksAgent,
        agent_id="zoho_books",
        agent_name="Zoho Books Agent"
    )
    app = _server.app
    
except Exception as e:
    import sys
    print(f"WARNING: BaseAgent ZohoBooksAgent failed: {e}", file=sys.stderr)
    
    # Fallback to legacy if available
    if legacy_available:
        from .zoho_books_agent import ZohoBooksAgent
        # Create a simple FastAPI wrapper for legacy agent
        from fastapi import FastAPI
        app = FastAPI(title="Zoho Books Agent - Legacy Mode")
        
        @app.get("/health")
        async def health():
            return {"status": "healthy", "agent": "zoho_books", "mode": "legacy"}
        
        @app.post("/execute")
        async def execute(request):
            return {"status": "error", "error": "Legacy mode - use /metrics endpoint directly"}
    else:
        raise RuntimeError("Neither BaseAgent nor Legacy ZohoBooksAgent available")

__all__ = ['ZohoBooksAgent', 'app']
