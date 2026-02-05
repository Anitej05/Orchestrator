# agents/spreadsheet_agent.py
# WRAPPER - Runs the spreadsheet agent package

"""
This wrapper module initializes the spreadsheet agent from the spreadsheet_agent/ package.
When run directly, it starts the FastAPI server on port 9000.
"""

import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Setup paths for standalone execution
SCRIPT_DIR = Path(__file__).parent.absolute()  # agents/ directory
BACKEND_DIR = SCRIPT_DIR.parent  # backend/ directory
PACKAGE_DIR = SCRIPT_DIR / "spreadsheet_agent"  # agents/spreadsheet_agent/

# Ensure paths are in sys.path for imports
# Order matters: PROJECT_ROOT > BACKEND_DIR > SCRIPT_DIR > PACKAGE_DIR
# We insert in reverse order.

if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# BACKEND_DIR (allows 'import services', 'import agents')
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# PROJECT_ROOT (allows 'import backend.services')
PROJECT_ROOT = BACKEND_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# Import the agent from the package
try:
    logger.info("📦 Importing Spreadsheet Agent...")
    # When run standalone, import from the local package
    from spreadsheet_agent import app as agent_app
    app = agent_app
    logger.info("✅ Successfully imported Spreadsheet Agent")
except ImportError as e:
    logger.error(f"❌ Failed to import spreadsheet_agent: {e}")
    
    # Fallback - try with different import path
    try:
        from agents.spreadsheet_agent import app as agent_app
        app = agent_app
        logger.info("✅ Imported via agents.spreadsheet_agent")
    except ImportError:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        
        # Fallback app
        app = FastAPI(title="Spreadsheet Agent (Fallback)")
        
        @app.get("/health")
        async def fallback_health():
            return {"status": "error", "message": f"Import failed: {e}"}
        
        raise RuntimeError(f"Failed to import spreadsheet_agent: {e}")

# Add CORS middleware if not already present
try:
    if not any(middleware.cls.__name__ == "CORSMiddleware" for middleware in app.user_middleware):
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
except Exception:
    pass  # CORS already set up

# -------------------------
# If run directly
# -------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv('SPREADSHEET_AGENT_PORT', 9000))
    logger.info(f"🚀 Starting Spreadsheet Agent on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
