"""
Browser Agent - AgentServer Entry Point

UAP-compliant browser automation agent using BaseAgent architecture.
Delegates to base_agent_impl.BrowserAgent via AgentServer.
"""

# CRITICAL: Load .env BEFORE any imports that trigger InferenceService/KeyManager singletons.
from pathlib import Path as _Path
from dotenv import load_dotenv as _load_dotenv
_load_dotenv(_Path(__file__).parent.parent.parent / ".env")

import os
import sys
from pathlib import Path

# ==================== ROBUST PATH HANDLING ====================
PACKAGE_DIR = Path(__file__).parent.absolute()
AGENTS_DIR = PACKAGE_DIR.parent
BACKEND_DIR = AGENTS_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

for path in [str(PROJECT_ROOT), str(BACKEND_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from backend.utils.mega_logger import setup_mega_logger
logger = setup_mega_logger("BrowserAgent")

# ============================================================================
# BASEAGENT SERVER (primary)
# ============================================================================

try:
    from .base_agent_impl import BrowserAgent as BaseBrowserAgent
    from backend.base_agent.server import create_agent_server

    _server = create_agent_server(
        agent_class=BaseBrowserAgent,
        agent_id="browser_agent",
        agent_name="Browser Automation Agent"
    )
    app = _server.app
    logger.info("BaseAgent Browser implementation loaded successfully")

except Exception as e:
    logger.error(f"Failed to initialize BaseAgent Browser implementation: {e}")
    import traceback
    traceback.print_exc()

    _init_error = str(e)

    from fastapi import FastAPI
    app = FastAPI(title="Browser Agent (error)")

    @app.get("/health")
    async def health():
        return {"status": "unhealthy", "error": "Failed to load Browser Agent", "details": _init_error}


# ============================================================================
# STANDALONE RUNNER
# ============================================================================

def run_agent() -> None:
    """Run Browser Agent as standalone service on port 8090"""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)


if __name__ == "__main__":
    run_agent()
