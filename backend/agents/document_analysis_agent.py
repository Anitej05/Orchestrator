# agents/document_analysis_agent.py
# SOTA WRAPPER - Delegates to agents/document_agent/ modularized components

from fastapi import FastAPI, HTTPException
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

log_file_path = Path(__file__).parent / "document_agent.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file_path, mode='w')
    ]
)
logger = logging.getLogger(__name__)
load_dotenv()

# ==================== ROBUST PATH HANDLING ====================
# Handle various execution contexts (main.py, direct run, etc.)

SCRIPT_DIR = Path(__file__).parent.absolute()  # agents/ directory
BACKEND_DIR = SCRIPT_DIR.parent  # backend/ directory

# Ensure paths are in sys.path for imports
# IMPORTANT: Add BACKEND_DIR first so backend/schemas.py is found before document_agent/schemas.py
# DO NOT add DOCUMENT_AGENT_DIR as it causes import conflicts with schemas module
for path in [str(BACKEND_DIR), str(SCRIPT_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

logger.info(f"🔧 Path setup: script_dir={SCRIPT_DIR}, backend_dir={BACKEND_DIR}")
logger.info(f"🔧 sys.path includes: backend/, agents/")

# Import new modularized Document Agent with proper error handling
app = None
try:
    logger.info("📦 Attempting to import document_agent module...")
    from document_agent import app as document_app
    app = document_app
    logger.info("✅ Successfully imported document_agent from local path")
except ImportError as e:
    logger.error(f"❌ Failed to import document_agent: {e}", exc_info=True)
    # Create fallback minimal app to prevent complete startup failure
    logger.warning("⚠️ Creating minimal fallback app")
    app = FastAPI(title="Document Agent (Fallback)")
    
    @app.get("/health")
    async def fallback_health():
        return {"status": "error", "message": "Import failed, check logs"}
    
    raise RuntimeError(
        f"Failed to import document_agent module. Check logs for details. Error: {e}"
    )

# -------------------------
# If run directly (development/production)
# -------------------------
if __name__ == "__main__":
    import uvicorn
    import socket
    # For production, run using your process manager / container and don't use reload=True.
    # Use 0.0.0.0 to bind to all interfaces for better compatibility
    port = int(os.getenv('DOCUMENT_AGENT_PORT', 8070))
    # --- Port availability check ---
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("0.0.0.0", port))
        sock.close()
    except OSError:
        logger.error(f"Port {port} is already in use. Please free the port or use a different one.")
        sys.exit(1)
    uvicorn.run("document_analysis_agent:app", host="0.0.0.0", port=port, reload=False)
