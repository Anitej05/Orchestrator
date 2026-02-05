"""
File loader utility for loading conversation-associated files from storage.
Handles documents, images, and other file types.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

STORAGE_BASE_DIR = "storage"
DOCUMENTS_DIR = os.path.join(STORAGE_BASE_DIR, "documents")
IMAGES_DIR = os.path.join(STORAGE_BASE_DIR, "images")


def ensure_storage_dirs():
    """Ensure all storage directories exist."""
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)


def load_file(file_path: str) -> Optional[str]:
    """
    Load a single file from storage.
    
    Args:
        file_path: Path to the file (relative to storage directory or absolute)
    
    Returns:
        File content as string, or None if file doesn't exist
    """
    ensure_storage_dirs()
    
    # Normalize path
    if not os.path.isabs(file_path):
        full_path = os.path.join(STORAGE_BASE_DIR, file_path)
    else:
        full_path = file_path
    
    # Check if file exists
    if not os.path.exists(full_path):
        logger.warning(f"File not found: {full_path}")
        return None
    
    try:
        # Read file content
        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        logger.info(f"Loaded file: {full_path} ({len(content)} bytes)")
        return content
        
    except Exception as e:
        logger.error(f"Error loading file {full_path}: {e}")
        return None


def load_files(file_references: List[str]) -> Dict[str, Optional[str]]:
    """
    Load multiple files from storage based on their paths.
    
    Args:
        file_references: List of file paths
    
    Returns:
        Dictionary mapping file path to content (or None if file not found)
    
    Example:
        file_references = [
            "documents/document1.txt",
            "images/image1.png"
        ]
        loaded_files = load_files(file_references)
    """
    loaded_files = {}
    
    for file_path in file_references:
        content = load_file(file_path)
        loaded_files[file_path] = content
    
    return loaded_files


def get_file_info(file_path: str) -> Optional[Dict]:
    """
    Get metadata about a file (size, type, exists, etc).
    
    Args:
        file_path: Path to the file
    
    Returns:
        Dictionary with file info, or None if file doesn't exist
    """
    ensure_storage_dirs()
    
    if not os.path.isabs(file_path):
        full_path = os.path.join(STORAGE_BASE_DIR, file_path)
    else:
        full_path = file_path
    
    if not os.path.exists(full_path):
        return None
    
    try:
        file_stats = os.stat(full_path)
        return {
            "path": file_path,
            "absolute_path": full_path,
            "exists": True,
            "size": file_stats.st_size,
            "is_file": os.path.isfile(full_path),
            "is_dir": os.path.isdir(full_path),
            "extension": os.path.splitext(full_path)[1],
        }
    except Exception as e:
        logger.error(f"Error getting file info for {full_path}: {e}")
        return None



async def build_file_context(loaded_files: Dict[str, Optional[str]], max_chars: int = 1000, cms_service=None) -> str:
    """
    Build a text context from loaded files for inclusion in LLM prompts.
    
    Args:
        loaded_files: Dictionary of file path -> content
        max_chars: Maximum characters to include per file (local fallback)
        cms_service: Optional ContentManagementService instance for offloading large files
    
    Returns:
        Formatted text context of all files
    """
    if not loaded_files:
        return ""
    
    context_parts = []
    
    for file_path, content in loaded_files.items():
        if content is None:
            context_parts.append(f"FILE: {file_path}\n[File not found or could not be read]\n---")
            continue
        
        # Check for massive content
        if cms_service and len(content) > 5000:
            try:
                # Offload to CMS
                # We use a standard naming convention
                safe_name = f"file_{os.path.basename(file_path)}_{int(time.time())}.txt"
                
                # Dynamic import for enums to avoid top-level circular deps
                from services.content_management_service import ContentSource, ContentType, ContentPriority, ProcessingTaskType, ProcessingStrategy

                # 1. Register
                content_meta = await cms_service.register_content(
                    content=content,
                    name=safe_name,
                    source=ContentSource.UPLOAD,
                    content_type=ContentType.DOCUMENT,
                    priority=ContentPriority.ephemeral,
                    tags=["uploaded_file", f"path:{file_path}"],
                    thread_id="global" # Files are often shared or global for now
                )
                
                # 2. Summarize
                process_result = await cms_service.process_large_content(
                    content_id=content_meta.id,
                    task_type=ProcessingTaskType.SUMMARIZE,
                    strategy=ProcessingStrategy.FAST # Fast summary for context
                )
                
                summary = process_result.final_output
                context_parts.append(
                    f"FILE: {file_path}\n"
                    f"[LARGE CONTENT OFFLOADED TO CMS - ID: {content_meta.id}]\n"
                    f"Summary: {summary}\n"
                    f"Instructions: Use 'query_file_content' tool (if available) or specific CMS actions to read details."
                    f"\n---"
                )
                logger.info(f"📚 Offloaded large file {file_path} to CMS ({len(content)} chars)")
                continue
                
            except Exception as e:
                logger.warning(f"Failed to offload file {file_path} to CMS: {e}")
                # Fallback to local truncation
        
        # Truncate content if too long (Fallback or if no CMS)
        truncated_content = content[:max_chars]
        if len(content) > max_chars:
            truncated_content += f"\n... (truncated, {len(content) - max_chars} more characters)"
        
        context_parts.append(f"FILE: {file_path}\n{truncated_content}\n---")
    
    return "\n".join(context_parts)



def get_conversation_files(thread_id: str) -> List[str]:
    """
    Get all files associated with a conversation (by thread_id pattern).
    
    Args:
        thread_id: The conversation thread ID
    
    Returns:
        List of file paths
    """
    ensure_storage_dirs()
    
    file_references = []
    
    # Look for files with thread_id in their path
    for root, dirs, files in os.walk(STORAGE_BASE_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            # Check if thread_id appears in path (simple pattern matching)
            if thread_id in file_path:
                # Store relative path
                rel_path = os.path.relpath(file_path, STORAGE_BASE_DIR)
                file_references.append(rel_path)
    
    return file_references


if __name__ == "__main__":
    # Test file loader
    ensure_storage_dirs()
    
    # Create test files
    test_doc_path = os.path.join(DOCUMENTS_DIR, "test.txt")
    with open(test_doc_path, 'w') as f:
        f.write("This is a test document content.")
    
    # Test loading
    files = load_files(["documents/test.txt"])
    print("Loaded files:", files)
    
    # Test context building
    context = build_file_context(files)
    print("\nFile context:")
    print(context)
