"""
Integrations Agent - Universal Composio Integration Handler
BaseAgent-compliant implementation for any Composio-powered app.

Features:
  - In-chat OAuth: returns auth URL when connection is missing
  - App detection: identifies which Composio app a task requires
  - Session persistence: preserves context across conversations
  - Dynamic tool discovery via Composio SDK

Version: 2.0.0
"""

__version__ = "2.0.0"

import os
import sys
import logging
from pathlib import Path

# ==================== ROBUST PATH HANDLING ====================
PACKAGE_DIR = Path(__file__).parent.absolute()
AGENTS_DIR = PACKAGE_DIR.parent
BACKEND_DIR = AGENTS_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

for path in [str(PROJECT_ROOT), str(BACKEND_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Public API
from .service import IntegrationsAgentService
from .session_manager import ComposioSessionManager, get_session_manager
from .app_detector import AppDetector, get_app_detector
from .base_agent_impl import IntegrationsAgent, IntegrationsAgentConfig, get_agent

# ============================================================================
# MODULE-LEVEL APP EXPOSURE (required by uvicorn / AgentManager)
# ============================================================================

try:
    from backend.agents.base.server import create_agent_server

    _server = create_agent_server(
        agent_class=IntegrationsAgent,
        agent_id="integrations_agent",
        agent_name="Integrations Agent"
    )
    app = _server.app

except Exception as _e:
    import traceback
    traceback.print_exc()

    from fastapi import FastAPI
    app = FastAPI(title="Integrations Agent (error)")

    @app.get("/health")
    async def health():
        return {"status": "unhealthy", "error": "Failed to load Integrations Agent", "details": str(_e)}


# ============================================================================
# STANDALONE RUNNER
# ============================================================================

def run_agent() -> None:
    """Run the Integrations Agent server on port 8075."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8075)


if __name__ == "__main__":
    run_agent()
