"""
Mail Agent - AgentServer Entry Point

Legacy Gmail assistant with LLM-powered email understanding.
Uses BaseAgent architecture via AgentServer.
Note: Prefer gmail_agent for new tasks — this agent is retained for backward compatibility.
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
logger = setup_mega_logger("MailAgent")

# ============================================================================
# BASEAGENT SERVER (primary)
# ============================================================================

try:
    from .base_agent_impl import MailAgent as BaseMailAgent
    from backend.agents.base.server import create_agent_server

    _server = create_agent_server(
        agent_class=BaseMailAgent,
        agent_id="mail_agent",
        agent_name="Mail Agent"
    )
    app = _server.app
    logger.info("BaseAgent Mail implementation loaded successfully")

except Exception as e:
    logger.error(f"Failed to initialize BaseAgent Mail implementation: {e}")
    import traceback
    traceback.print_exc()

    _init_error = str(e)

    # Fallback: try legacy app
    try:
        from .agent import app as legacy_app
        app = legacy_app
        logger.info("Fell back to legacy Mail Agent app")
    except Exception:
        from fastapi import FastAPI
        app = FastAPI(title="Mail Agent (error)")

        @app.get("/health")
        async def health():
            return {"status": "unhealthy", "error": "Failed to load Mail Agent", "details": _init_error}


# ============================================================================
# STANDALONE RUNNER
# ============================================================================

def run_agent() -> None:
    """Run Mail Agent as standalone service on port 8040"""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8040)


if __name__ == "__main__":
    run_agent()
