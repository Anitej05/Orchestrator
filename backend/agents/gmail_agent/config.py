# agents/gmail_agent/config.py
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# API Keys
COMPOSIO_API_KEY = os.getenv("COMPOSIO_API_KEY")

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-70b-versatile")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

# Storage
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
ATTACHMENT_DIR = PROJECT_ROOT / "storage" / "gmail_agent" / "attachments"
ATTACHMENT_TTL_HOURS = 72

# Gmail Agent Settings
MAX_SEARCH_RESULTS = 50
DEFAULT_PAGE_SIZE = 10
MAX_CONCURRENT_FETCHES = 5
BATCH_SIZE = 10

# Logging
logger = logging.getLogger("gmail_agent")
logger.setLevel(logging.INFO)

# Ensure attachment directory exists
ATTACHMENT_DIR.mkdir(parents=True, exist_ok=True)
