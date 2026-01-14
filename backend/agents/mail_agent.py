# agents/mail_agent.py
# WRAPPER - Delegates to agents/mail_agent modularized components

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# ==================== ROBUST PATH HANDLING ====================
SCRIPT_DIR = Path(__file__).parent.absolute()
BACKEND_DIR = SCRIPT_DIR.parent

# Ensure agents and backend are in sys.path
for path in [str(SCRIPT_DIR), str(BACKEND_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

logger.info(f"🔧 Path setup: script_dir={SCRIPT_DIR}, backend_dir={BACKEND_DIR}")

# Import modularized Mail Agent from the package
# We use absolute import from agents to avoid circularity if the file is named mail_agent.py
app = None
try:
    logger.info("📦 Attempting to import mail_agent package logic...")
    # This imports from agents/mail_agent/agent.py
    from agents.mail_agent.agent import app as mail_app
    app = mail_app
    logger.info("✅ Successfully imported mail_agent from package")
except ImportError as e:
    logger.error(f"❌ Failed to import mail_agent package: {e}", exc_info=True)
    # Create fallback minimal app
    app = FastAPI(title="Mail Agent (Fallback)")
    
    @app.get("/health")
    async def fallback_health():
        return {"status": "error", "message": f"Import failed: {e}"}
    
    raise RuntimeError(f"Failed to import mail_agent package: {e}")

# Add CORS middleware
if not any(middleware.cls.__name__ == "CORSMiddleware" for middleware in app.user_middleware):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

if __name__ == "__main__":
    import uvicorn
    # Discovery by main.py looks for port in the file or uses default
    port = int(os.getenv('MAIL_AGENT_PORT', 8040))
    logger.info(f"Starting Mail Agent Wrapper on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
