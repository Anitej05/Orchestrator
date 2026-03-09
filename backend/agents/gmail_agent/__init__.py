# agents/gmail_agent/__init__.py
"""
Gmail Agent - BaseAgent implementation for Gmail operations

Uses Composio Gmail tools and BaseAgent framework for clean architecture.
"""

import os
import sys
import logging
from pathlib import Path

# ==================== ROBUST PATH HANDLING ====================
PACKAGE_DIR = Path(__file__).parent.absolute()
AGENTS_DIR = PACKAGE_DIR.parent
BACKEND_DIR = AGENTS_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

# Ensure correct paths in sys.path
for path in [str(PROJECT_ROOT), str(BACKEND_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

__version__ = "1.0.0"

# Configure logging via centralized Mega Logger
from backend.utils.mega_logger import setup_mega_logger
logger = setup_mega_logger("GmailAgent")

# ============================================================================
# BASEAGENT IMPLEMENTATION
# ============================================================================

try:
    from .base_agent_impl import GmailAgent as BaseGmailAgent
    from backend.agents.base.server import create_agent_server
    
    logger.info("Initializing BaseAgent Gmail implementation...")
    _server = create_agent_server(
        agent_class=BaseGmailAgent,
        agent_id="gmail_agent",
        agent_name="Gmail Agent"
    )
    app = _server.app
    logger.info("BaseAgent Gmail implementation loaded successfully")
    
except Exception as e:
    logger.error(f"Failed to initialize BaseAgent Gmail implementation: {e}")
    import traceback
    traceback.print_exc()

    # Capture error before `e` goes out of scope (Python 3 clears it after except block)
    _init_error = str(e)

    # Create minimal health-check app as fallback
    from fastapi import FastAPI
    app = FastAPI(title="Gmail Agent (error)")

    @app.get("/health")
    async def health():
        return {"status": "unhealthy", "error": "Failed to load Gmail Agent", "details": _init_error}


# ============================================================================
# STANDALONE RUNNER
# ============================================================================

def run_agent() -> None:
    """Run Gmail Agent as standalone service on port 8003"""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)


if __name__ == "__main__":
    run_agent()
