"""
PPT Agent

Presentation processing agent powered by ReAct loop.
True subclass of BaseAgent.
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

# Ensure correct paths in sys.path
for path in [str(PROJECT_ROOT), str(BACKEND_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from backend.base_agent.server import create_agent_server
from .base_agent_impl import PPTAgent

from backend.utils.mega_logger import setup_mega_logger
logger = setup_mega_logger("PPTAgent")

_server = create_agent_server(
    agent_class=PPTAgent,
    agent_id="ppt_agent",
    agent_name="PPT Agent"
)
app = _server.app  # Export the FastAPI instance, NOT the AgentServer wrapper

def run_agent() -> None:
    import uvicorn
    port = int(os.getenv("PPT_AGENT_PORT", "8056"))
    logger.info(f"Starting PPT Agent server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    run_agent()
