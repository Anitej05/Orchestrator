# agents/spreadsheet_agent_agent.py
# WRAPPER - Delegates to agents/spreadsheet_agent/ modularized components

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
# Handle various execution contexts (main.py, direct run, etc.)

SCRIPT_DIR = Path(__file__).parent.absolute()  # agents/ directory
BACKEND_DIR = SCRIPT_DIR.parent  # backend/ directory
SPREADSHEET_AGENT_DIR = SCRIPT_DIR / "spreadsheet_agent"  # agents/spreadsheet_agent/

# Ensure paths are in sys.path for imports
for path in [str(SCRIPT_DIR), str(BACKEND_DIR), str(SPREADSHEET_AGENT_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

logger.info(f"🔧 Path setup: script_dir={SCRIPT_DIR}, backend_dir={BACKEND_DIR}")
logger.info(f"🔧 sys.path includes: agents/, backend/, spreadsheet_agent/")

# Import new modularized Spreadsheet Agent with proper error handling
app = None
try:
    logger.info("📦 Attempting to import spreadsheet_agent module...")
    from spreadsheet_agent.main import app as spreadsheet_app
    app = spreadsheet_app
    logger.info("✅ Successfully imported spreadsheet_agent from local path")
except ImportError as e:
    logger.error(f"❌ Failed to import spreadsheet_agent: {e}", exc_info=True)
    # Create fallback minimal app to prevent complete startup failure
    logger.warning("⚠️ Creating minimal fallback app")
    app = FastAPI(title="Spreadsheet Agent (Fallback)")
    
    @app.get("/health")
    async def fallback_health():
        return {"status": "error", "message": "Import failed, check logs"}
    
    raise RuntimeError(
        f"Failed to import spreadsheet_agent module. Check logs for details. Error: {e}"
    )

# Add CORS middleware if not already present
if not any(middleware.cls.__name__ == "CORSMiddleware" for middleware in app.user_middleware):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# -------------------------
# If run directly (development/production)
# -------------------------
if __name__ == "__main__":
    import uvicorn
    # For production, run using your process manager / container and don't use reload=True.
    # Use 0.0.0.0 to bind to all interfaces for better compatibility
    port = int(os.getenv('SPREADSHEET_AGENT_PORT', 8041))
    uvicorn.run("spreadsheet_agent_agent:app", host="0.0.0.0", port=port, reload=False)
