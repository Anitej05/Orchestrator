"""
Configuration and constants for Spreadsheet Agent
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ============== PATHS ==============
# Use root storage directory as specified
ROOT_DIR = Path(__file__).parent.parent.parent.parent  # Get to repo root
STORAGE_DIR = ROOT_DIR / "storage" / "spreadsheet_agent"
SESSIONS_DIR = STORAGE_DIR / "sessions"
MEMORY_CACHE_DIR = STORAGE_DIR / "memory"

# Create directories
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
MEMORY_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ============== LLM CONFIGURATION ==============
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# LLM Models - Priority order: Cerebras → Groq → NVIDIA → Google
# REVERTED to tested working models (not llama-3.1 which has JSON parsing issues)
GROQ_MODEL = "openai/gpt-oss-120b"  # Tested working model - avoids JSON parsing failures
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

CEREBRAS_MODEL = "gpt-oss-120b"  # Tested working model - consistently produces valid JSON
CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"

NVIDIA_MODEL = "meta/llama-3.1-70b-instruct"  # NVIDIA NIM
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

GOOGLE_MODEL = "gemini-1.5-flash"  # Fast and efficient
GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

OPENAI_MODEL = "gpt-4o-mini"  # Cost-effective fallback
OPENAI_BASE_URL = "https://api.openai.com/v1"

ANTHROPIC_MODEL = "claude-3-5-haiku-20241022"  # Fast and affordable
ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"

# LLM Settings
LLM_TEMPERATURE = 0.1  # Hardcoded - do NOT use config for consistency with old agent
LLM_MAX_TOKENS_QUERY = 2000  # Consistent with old agent
LLM_MAX_TOKENS_QUERY = 2000
LLM_MAX_TOKENS_CODE_GEN = 2000
LLM_TIMEOUT = 60

# ============== AGENT CONFIGURATION ==============
AGENT_ID = "spreadsheet_agent"
AGENT_PORT = int(os.getenv("SPREADSHEET_AGENT_PORT", 8041))

# ============== FILE MANAGEMENT ==============
MAX_FILE_SIZE_MB = 50
ALLOWED_EXTENSIONS = [".csv", ".xlsx", ".xls"]
DEFAULT_TTL_HOURS = None  # No expiration by default
AUTO_CLEANUP = True
CLEANUP_INTERVAL_HOURS = 24

# ============== DISPLAY SETTINGS ==============
MAX_ROWS_DISPLAY = 100
MAX_COLUMNS_DISPLAY = 50
PREVIEW_ROWS = 5

# ============== QUERY SETTINGS ==============
MAX_QUERY_ITERATIONS = 5
MAX_RETRIES = 3
BACKOFF_FACTOR = 2

# ============== MEMORY/CACHE SETTINGS ==============
MEMORY_CACHE_MAX_SIZE = 1000  # Max number of cache entries
MEMORY_CACHE_TTL_SECONDS = 3600  # 1 hour
CONTEXT_MEMORY_MAX_TOKENS = 2000  # Max tokens for context

# ============== SESSION SETTINGS ==============
SESSION_TIMEOUT_HOURS = 24
MAX_OPERATIONS_IN_CONTEXT = 5
MAX_CONVERSATION_TURNS = 10

# ============== LOGGING ==============
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
