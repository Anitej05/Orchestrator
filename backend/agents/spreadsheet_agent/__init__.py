"""
Spreadsheet Agent v3.0

Unified spreadsheet operations with LLM-powered task decomposition.
Uses BaseAgent architecture via AgentServer.
"""

import os
import sys
from pathlib import Path

# CRITICAL: Load .env BEFORE any imports that trigger InferenceService/KeyManager singletons.
from dotenv import load_dotenv
_BACKEND_DIR = Path(__file__).parent.parent.parent
load_dotenv(_BACKEND_DIR / ".env")

# ==================== ROBUST PATH HANDLING ====================
PACKAGE_DIR = Path(__file__).parent.absolute()
AGENTS_DIR = PACKAGE_DIR.parent
BACKEND_DIR = AGENTS_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

for path in [str(PROJECT_ROOT), str(BACKEND_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from backend.utils.mega_logger import setup_mega_logger
logger = setup_mega_logger("SpreadsheetAgent")

from .config import AGENT_PORT, AGENT_VERSION

# ============================================================================
# BASEAGENT SERVER (primary)
# ============================================================================

try:
    from .base_agent_impl import SpreadsheetAgent as BaseSpreadsheetAgent
    from backend.base_agent.server import create_agent_server

    _server = create_agent_server(
        agent_class=BaseSpreadsheetAgent,
        agent_id="spreadsheet_agent",
        agent_name="Spreadsheet Agent"
    )
    app = _server.app
    logger.info(f"BaseAgent Spreadsheet implementation loaded (v{AGENT_VERSION})")

except Exception as e:
    logger.error(f"Failed to initialize BaseAgent Spreadsheet implementation: {e}")
    import traceback
    traceback.print_exc()

    _init_error = str(e)

    from fastapi import FastAPI
    app = FastAPI(title="Spreadsheet Agent (error)")

    @app.get("/health")
    async def health():
        return {"status": "unhealthy", "error": "Failed to load Spreadsheet Agent", "details": _init_error}


# ============================================================================
# STANDALONE RUNNER
# ============================================================================

def run_agent() -> None:
    """Run Spreadsheet Agent as standalone service."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=AGENT_PORT)


if __name__ == "__main__":
    run_agent()
