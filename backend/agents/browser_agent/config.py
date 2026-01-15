"""
Browser Agent - Configuration

Centralized configuration for the browser agent.
"""

import os
from pathlib import Path
from typing import Optional

# Determine project root (2 levels up from browser_agent folder)
_CURRENT_FILE = Path(__file__).resolve()
_BROWSER_AGENT_DIR = _CURRENT_FILE.parent
_AGENTS_DIR = _BROWSER_AGENT_DIR.parent
_BACKEND_DIR = _AGENTS_DIR.parent
_PROJECT_ROOT = _BACKEND_DIR.parent

# Storage configuration - Use project root storage folder
STORAGE_ROOT = _PROJECT_ROOT / "storage" / "browser_agent"

class BrowserAgentConfig:
    """Centralized configuration for browser agent"""
    
    # Storage root
    STORAGE_ROOT: Path = STORAGE_ROOT
    
    # Storage paths - all relative to browser_agent storage root
    DOWNLOADS_DIR: Path = STORAGE_ROOT / "downloads"
    UPLOADS_DIR: Path = STORAGE_ROOT / "uploads"
    SCREENSHOTS_DIR: Path = STORAGE_ROOT / "screenshots"
    
    # Timeouts (in milliseconds)
    NAVIGATION_TIMEOUT: int = 60000  # 60 seconds
    SCREENSHOT_TIMEOUT: int = 15000  # 15 seconds
    CLICK_TIMEOUT: int = 10000  # 10 seconds
    TYPE_TIMEOUT: int = 10000  # 10 seconds
    
    # Limits
    MAX_STEPS: int = 50
    MAX_RETRIES: int = 3
    MAX_HISTORY_ITEMS: int = 50
    MAX_CONSECUTIVE_FAILURES: int = 5
    
    # File management
    FILE_TTL_HOURS: int = 72  # Auto-cleanup after 72 hours
    
    # LLM settings
    LLM_TIMEOUT: int = 60  # seconds
    
    def __init__(self):
        """Initialize and ensure directories exist"""
        self.DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
        self.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        self.SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    def get_download_path(self, filename: Optional[str] = None) -> Path:
        """Get path for downloading files"""
        if filename:
            return self.DOWNLOADS_DIR / filename
        return self.DOWNLOADS_DIR
    
    def get_upload_path(self, filename: str) -> Optional[Path]:
        """Get path for file to upload (checks if exists)"""
        # First check in uploads directory
        upload_path = self.UPLOADS_DIR / filename
        if upload_path.exists():
            return upload_path
        
        # Check if it's an absolute path
        abs_path = Path(filename)
        if abs_path.is_absolute() and abs_path.exists():
            return abs_path
            
        # Check relative to project root (for paths like storage/documents/...)
        project_rel_path = _PROJECT_ROOT / filename
        if project_rel_path.exists():
            return project_rel_path
        
        return None
    
    def get_screenshot_path(self, filename: Optional[str] = None) -> Path:
        """Get path for screenshots"""
        if filename:
            return self.SCREENSHOTS_DIR / filename
        return self.SCREENSHOTS_DIR
    
    def list_available_uploads(self) -> list[str]:
        """List files available for upload"""
        if self.UPLOADS_DIR.exists():
            return [f.name for f in self.UPLOADS_DIR.iterdir() if f.is_file()]
        return []


# Global config instance
CONFIG = BrowserAgentConfig()
