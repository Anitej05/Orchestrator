"""
File Server Utility - Secure file lookup and serving for agent-generated content.

This module provides utilities for securely serving files from the backend storage directory
with path traversal protection, MIME type detection, and smart filename resolution.

Security Features:
- Path traversal prevention (blocks ../ attempts)
- Validates all resolved paths stay within backend/storage
- Requires Clerk JWT authentication
- Safe error handling without information leakage

Storage Structure:
- backend/storage/documents/ - Generated/modified documents
- backend/storage/document_versions/ - Document version history
- backend/storage/content/ - Generic content files
- backend/storage/images/ - Generated images
- backend/storage/spreadsheets/ - Excel/CSV files
- Plus other dynamic agent folders
"""

import os
import mimetypes
import logging
from pathlib import Path
from typing import Optional, Tuple, List
from datetime import datetime

logger = logging.getLogger(__name__)

# Base storage directory (absolute path for security)
STORAGE_BASE = Path(__file__).parent.parent / "storage"
STORAGE_BASE = STORAGE_BASE.resolve()  # Resolve to absolute path

# Priority order for filename search (most common first)
SEARCH_FOLDERS = [
    "documents",
    "content",
    "images",
    "spreadsheets",
    "document_versions",
    "browser_downloads",
    "gmail_attachments",
    "artifacts",
]


def is_safe_path(file_path: str) -> bool:
    """
    Validate that the file path doesn't contain path traversal attempts.
    
    Args:
        file_path: The file path to validate
        
    Returns:
        True if path is safe, False otherwise
    """
    # Reject paths with path traversal patterns
    dangerous_patterns = ["..", "~", "//", "\\\\"]
    for pattern in dangerous_patterns:
        if pattern in file_path:
            logger.warning(f"Path traversal attempt detected: {file_path}")
            return False
    
    # Reject absolute paths (should be relative to storage)
    if os.path.isabs(file_path):
        logger.warning(f"Absolute path rejected: {file_path}")
        return False
    
    return True


def resolve_and_validate_path(file_path: str) -> Optional[Path]:
    """
    Resolve a file path and validate it stays within STORAGE_BASE.
    
    Args:
        file_path: Relative path from storage root
        
    Returns:
        Resolved absolute Path object if valid, None otherwise
    """
    try:
        # Construct full path
        full_path = (STORAGE_BASE / file_path).resolve()
        
        # Verify it's within storage base (prevents escaping via symlinks)
        if not str(full_path).startswith(str(STORAGE_BASE)):
            logger.error(f"Path escape attempt: {file_path} -> {full_path}")
            return None
        
        return full_path
    
    except Exception as e:
        logger.error(f"Path resolution error for {file_path}: {e}")
        return None


def find_file_in_storage(filename: str) -> Optional[Tuple[Path, str]]:
    """
    Smart file lookup across storage folders. If only a filename is provided,
    searches in priority order and returns the newest version if found in multiple locations.
    
    Args:
        filename: Just the filename (e.g., "report.pdf") or relative path
        
    Returns:
        Tuple of (resolved_path, relative_path_from_storage) or None if not found
    """
    # If it looks like a path with folders, try direct lookup first
    if "/" in filename or "\\" in filename:
        full_path = resolve_and_validate_path(filename.replace("\\", "/"))
        if full_path and full_path.exists() and full_path.is_file():
            relative = full_path.relative_to(STORAGE_BASE)
            return (full_path, str(relative))
    
    # Search across common folders
    found_files: List[Tuple[Path, datetime]] = []
    
    for folder in SEARCH_FOLDERS:
        search_path = STORAGE_BASE / folder / filename
        if search_path.exists() and search_path.is_file():
            # Verify it's still within bounds (paranoid check)
            if str(search_path.resolve()).startswith(str(STORAGE_BASE)):
                mod_time = datetime.fromtimestamp(search_path.stat().st_mtime)
                found_files.append((search_path, mod_time))
                logger.info(f"Found {filename} in {folder} (modified: {mod_time})")
    
    # Return newest version if multiple found
    if found_files:
        newest = max(found_files, key=lambda x: x[1])
        newest_path = newest[0]
        relative = newest_path.relative_to(STORAGE_BASE)
        logger.info(f"Selected newest version: {relative}")
        return (newest_path, str(relative))
    
    # Not found anywhere
    logger.warning(f"File not found in any storage location: {filename}")
    return None


def get_mime_type(file_path: Path) -> str:
    """
    Detect MIME type from file extension with safe fallbacks.
    
    Args:
        file_path: Path to the file
        
    Returns:
        MIME type string
    """
    # Initialize mimetypes if needed
    if not mimetypes.inited:
        mimetypes.init()
    
    mime_type, _ = mimetypes.guess_type(str(file_path))
    
    # Fallback for common types
    if not mime_type:
        ext = file_path.suffix.lower()
        mime_map = {
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".csv": "text/csv",
            ".json": "application/json",
            ".txt": "text/plain",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".svg": "image/svg+xml",
        }
        mime_type = mime_map.get(ext, "application/octet-stream")
    
    return mime_type


def should_inline_preview(mime_type: str) -> bool:
    """
    Determine if file should be previewed inline or downloaded.
    
    Args:
        mime_type: MIME type of the file
        
    Returns:
        True if should use Content-Disposition: inline
    """
    # Types that browsers can preview
    inline_types = [
        "application/pdf",
        "text/plain",
        "text/html",
        "text/csv",
        "application/json",
        "image/",  # All image types
    ]
    
    for inline_type in inline_types:
        if mime_type.startswith(inline_type):
            return True
    
    return False
