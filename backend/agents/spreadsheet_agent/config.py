"""
Spreadsheet Agent v3.0 - Configuration

Simplified configuration for the redesigned agent.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# .env lives in backend/ — resolve the path so it works regardless of cwd
_backend_dir = Path(__file__).resolve().parent.parent.parent
load_dotenv(_backend_dir / ".env")

# ============================================================================
# PATHS
# ============================================================================

ROOT_DIR = Path(__file__).parent.parent.parent.parent  # Get to repo root
STORAGE_DIR = ROOT_DIR / "storage" / "spreadsheet_agent"
AGENT_DIR = Path(__file__).parent

# Create directories
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# AGENT CONFIGURATION
# ============================================================================

AGENT_ID = "spreadsheet_agent"
AGENT_PORT = int(os.getenv("SPREADSHEET_AGENT_PORT", 9000))
AGENT_VERSION = "3.0.0"

# ============================================================================
# FILE HANDLING
# ============================================================================

MAX_FILE_SIZE_MB = 100
ALLOWED_EXTENSIONS = [".csv", ".xlsx", ".xls"]
LARGE_FILE_THRESHOLD_MB = 50  # Files larger than this use chunked processing

# ============================================================================
# LLM CONFIGURATION
# ============================================================================

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Provider configurations (same order as mail agent: Cerebras → NVIDIA → Groq)
# API keys are loaded from .env — NEVER hardcode keys in source code.
CEREBRAS_KEYS = [k.strip() for k in (os.getenv("CEREBRAS_API_KEYS", "") or "").split(",") if k.strip()]
if not CEREBRAS_KEYS and CEREBRAS_API_KEY:
    CEREBRAS_KEYS = [CEREBRAS_API_KEY]

LLM_PROVIDERS = []
# Add one provider entry per Cerebras key for round-robin
for _key in CEREBRAS_KEYS:
    LLM_PROVIDERS.append({
        "name": "cerebras",
        "api_key": _key,
        "model": "gpt-oss-120b",
        "summary_model": "llama-3.3-70b",
        "base_url": "https://api.cerebras.ai/v1"
    })
# Fallback providers
LLM_PROVIDERS.extend([
    {
        "name": "nvidia",
        "api_key": NVIDIA_API_KEY,
        "model": "minimaxai/minimax-m2",
        "summary_model": "llama-3.1-405b-instruct",
        "base_url": "https://integrate.api.nvidia.com/v1"
    },
    {
        "name": "groq",
        "api_key": GROQ_API_KEY,
        "model": "openai/gpt-oss-120b",
        "summary_model": "llama-3.3-70b-versatile",
        "base_url": "https://api.groq.com/openai/v1"
    }
])


LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 4000
LLM_TIMEOUT = 15  # Reduced from 60 to allow provider failover within orchestrator limit

# ============================================================================
# CACHING
# ============================================================================

CACHE_MAX_SIZE = 100  # Maximum cached DataFrames
CACHE_TTL_HOURS = 2   # Cache expiration
SESSION_TIMEOUT_HOURS = 24

# ============================================================================
# CONTEXT BUILDING
# ============================================================================

MAX_CONTEXT_TOKENS = 8000
MAX_SAMPLE_ROWS = 20
MAX_COLUMN_PREVIEW = 50

# ============================================================================
# LOGGING
# ============================================================================

import sys
backend_root = Path(__file__).resolve().parents[3]
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from backend.utils.mega_logger import setup_mega_logger
logger = setup_mega_logger("SpreadsheetAgent")
