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
STORAGE_DIR = ROOT_DIR / "storage" / "spreadsheets"
SESSIONS_DIR = ROOT_DIR / "storage" / "spreadsheet_sessions"
MEMORY_CACHE_DIR = ROOT_DIR / "storage" / "spreadsheet_memory"

# Create directories
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
MEMORY_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ============== LLM CONFIGURATION ==============
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")

# LLM Models
CEREBRAS_MODEL = "llama3.1-70b"
CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# LLM Settings
LLM_TEMPERATURE = 0.1
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
