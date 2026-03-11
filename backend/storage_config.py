"""
Centralized Storage Configuration

Single source of truth for all storage paths in the Orchestrator.
Every module that needs a storage path imports from here.
"""

from pathlib import Path

# ── Root Paths ────────────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = BACKEND_DIR.parent.resolve()
STORAGE_ROOT = PROJECT_ROOT / "storage"

# ── Content-Type Directories (replaces per-agent folders) ─────────────────────
DOCUMENTS_DIR = STORAGE_ROOT / "documents"
IMAGES_DIR = STORAGE_ROOT / "images"
SPREADSHEETS_DIR = STORAGE_ROOT / "spreadsheets"
SCREENSHOTS_DIR = STORAGE_ROOT / "screenshots"
DOWNLOADS_DIR = STORAGE_ROOT / "downloads"
TEMP_DIR = STORAGE_ROOT / "temp"
ATTACHMENTS_DIR = STORAGE_ROOT / "attachments"

# ── Internal System Directories ───────────────────────────────────────────────
SYSTEM_DIR = STORAGE_ROOT / "system"
ARTIFACTS_DIR = STORAGE_ROOT / "artifacts"
VECTOR_STORE_DIR = STORAGE_ROOT / "vector_store"
ORCHESTRATOR_DIR = STORAGE_ROOT / "orchestrator"
SHARED_DIR = STORAGE_ROOT / "shared"
CONTENT_DIR = SYSTEM_DIR / "content"
PROFILES_DIR = STORAGE_ROOT / "profiles"

# ── Ensure all directories exist ──────────────────────────────────────────────
for _dir in [
    STORAGE_ROOT,
    DOCUMENTS_DIR,
    IMAGES_DIR,
    SPREADSHEETS_DIR,
    SCREENSHOTS_DIR,
    DOWNLOADS_DIR,
    TEMP_DIR,
    ATTACHMENTS_DIR,
    SYSTEM_DIR,
    ARTIFACTS_DIR,
    VECTOR_STORE_DIR,
    ORCHESTRATOR_DIR,
    SHARED_DIR,
    CONTENT_DIR,
    PROFILES_DIR,
]:
    _dir.mkdir(parents=True, exist_ok=True)
