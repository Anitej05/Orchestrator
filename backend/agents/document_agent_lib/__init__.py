"""
Document Agent

Full document analysis and editing agent powered by ReAct loop.
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

from backend.agents.base.server import create_agent_server
from .base_agent_impl import DocumentAgent

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = create_agent_server(
    agent_class=DocumentAgent,
    agent_id="document_agent",
    agent_name="Document Agent"
)

def run_agent() -> None:
    import uvicorn
    # Document agent typically ran on 8050 in the legacy setup
    port = int(os.getenv("DOCUMENT_AGENT_PORT", "8050"))
    logger.info(f"Starting Document Agent server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    run_agent()
