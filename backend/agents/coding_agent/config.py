"""
Coding Agent - Configuration

Settings for the OpenCode-powered coding agent.
"""

import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

ROOT_DIR = Path(__file__).parent.parent.parent.parent  # Repo root
STORAGE_DIR = ROOT_DIR / "storage" / "coding_agent"
AGENT_DIR = Path(__file__).parent

STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# AGENT CONFIGURATION
# ============================================================================

AGENT_ID = "coding_agent"
AGENT_PORT = int(os.getenv("CODING_AGENT_PORT", 8080))
AGENT_VERSION = "1.0.0"

# ============================================================================
# OPENCODE SERVER CONFIGURATION
# ============================================================================

# Port for the OpenCode headless server (separate from the agent's own port)
OPENCODE_SERVER_PORT = int(os.getenv("OPENCODE_SERVER_PORT", 4096))
OPENCODE_SERVER_HOST = os.getenv("OPENCODE_SERVER_HOST", "127.0.0.1")

# Optional HTTP basic auth for the OpenCode server
OPENCODE_SERVER_PASSWORD = os.getenv("OPENCODE_SERVER_PASSWORD", "")
OPENCODE_SERVER_USERNAME = os.getenv("OPENCODE_SERVER_USERNAME", "opencode")

# Default project directory for OpenCode to operate in
# When None, the agent will use the workspace path from the orchestrator
OPENCODE_PROJECT_DIR = os.getenv("OPENCODE_PROJECT_DIR", None)

# ============================================================================
# TIMEOUTS
# ============================================================================

# Max seconds to wait for OpenCode server to boot
OPENCODE_STARTUP_TIMEOUT = int(os.getenv("OPENCODE_STARTUP_TIMEOUT", 45))

# Max seconds to wait for a coding task to complete
OPENCODE_REQUEST_TIMEOUT = int(os.getenv("OPENCODE_REQUEST_TIMEOUT", 300))

# Health check polling interval in seconds
HEALTH_CHECK_INTERVAL = 0.5

# ============================================================================
# BEHAVIOR
# ============================================================================

# Whether read-only operations (review, explain) skip the approval gate
AUTO_APPROVE_READS = True

# Maximum number of files to include in diff output
MAX_DIFF_FILES = 20

# Maximum lines per file in diff output
MAX_DIFF_LINES_PER_FILE = 500

# ============================================================================
# LOGGING
# ============================================================================

import sys
backend_root = Path(__file__).resolve().parents[3]
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from backend.utils.mega_logger import setup_mega_logger
logger = setup_mega_logger("CodingAgent")
