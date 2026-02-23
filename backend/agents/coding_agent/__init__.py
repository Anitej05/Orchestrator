"""
Coding Agent v1.0

AI-powered coding agent backed by OpenCode's headless server.
Supports code generation, review, testing, and multi-file refactoring.

Exposes standard UAP endpoints via AgentServer:
  /health, /execute, /capabilities, /metrics
"""

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

import logging
from .config import AGENT_PORT, AGENT_VERSION, logger

# ============================================================================
# BASEAGENT SERVER (primary)
# ============================================================================

try:
    from .agent import CodingAgent
    from backend.agents.base.server import create_agent_server

    _server = create_agent_server(
        agent_class=CodingAgent,
        agent_id="coding_agent",
        agent_name="Coding Agent",
    )
    app = _server.app  # Export BaseAgent app as primary

except Exception as e:
    print(f"WARNING: BaseAgent CodingAgent failed to initialize: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()

    # Fallback: minimal FastAPI for health checks only
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(
        title="Coding Agent (Fallback)",
        version=AGENT_VERSION,
        description="Coding Agent - fallback mode (OpenCode not available)",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        return {
            "status": "degraded",
            "agent_id": "coding_agent",
            "error": str(e),
            "initialized": False,
        }

    @app.post("/execute")
    async def execute(request: dict = {}):
        return {
            "status": "error",
            "error_message": (
                "Coding agent is in fallback mode. "
                "Ensure OpenCode is installed: npm i -g opencode-ai@latest"
            ),
        }


# ============================================================================
# FOR RUNNING DIRECTLY
# ============================================================================

def run_agent() -> None:
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=AGENT_PORT)


if __name__ == "__main__":
    run_agent()
