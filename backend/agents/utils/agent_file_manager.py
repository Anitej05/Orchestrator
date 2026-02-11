"""
Agent File Manager - Robust, Future-Proof File Management for Agents

This module provides a comprehensive file management system for agents that:
1. Persists file metadata across restarts
2. Supports file verification and cleanup
3. Integrates with the orchestrator's unified content system
4. Provides consistent API across all agents
5. Handles errors gracefully
6. Supports async operations
7. Is extensible for future needs

Usage:
    from agent_file_manager import AgentFileManager
    
    # Initialize for your agent
    file_manager = AgentFileManager(
        agent_id="spreadsheet_agent",
        storage_dir="storage/spreadsheets"
    )
    
    # Register file
    metadata = await file_manager.register_file(content, filename)
    
    # Get file
    content = file_manager.get_file_content(file_id)
    
    # Verify file exists
    exists = file_manager.verify_file(file_id)
"""

import os
import uuid
import json
import hashlib
import logging
import mimetypes
import asyncio
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod

# CMS Integration
import sys
from pathlib import Path
# agent_file_manager is in backend/agents/utils
# root is backend/
backend_root = Path(__file__).parent.parent.parent.resolve()
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

# CMS Integration moved inside methods to avoid circular imports
# from backend.services.content_management_service import (
#     ContentManagementService,
#     ContentSource,
#     ContentType,
#     ContentPriority
# )


logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class FileStatus(str, Enum):
    """Status of a file in the system"""
    ACTIVE = "active"           # File is available and valid
    PROCESSING = "processing"   # File is being processed
    EXPIRED = "expired"         # File has expired
    DELETED = "deleted"         # File has been deleted
    ERROR = "error"             # File has an error


class FileType(str, Enum):
    """Types of files"""
    SPREADSHEET = "spreadsheet"
    DOCUMENT = "document"
    IMAGE = "image"
    DATA = "data"
    CODE = "code"
    ARCHIVE = "archive"
    SCREENSHOT = "screenshot"
    DOWNLOAD = "download"
    ATTACHMENT = "attachment"
    OTHER = "other"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class AgentFileMetadata:
    """
    Comprehensive metadata for files managed by agents.
    Designed to be future-proof with extensible fields.
    """
    # Identity
    file_id: str
    agent_id: str
    original_name: str
    
    # Storage
    storage_path: str
    size_bytes: int
    checksum: str
    mime_type: str
    file_type: FileType
    
    # Status
    status: FileStatus = FileStatus.ACTIVE
    
    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    accessed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    expires_at: Optional[str] = None
    
    # Usage tracking
    access_count: int = 0
    
    # Orchestrator integration
    orchestrator_content_id: Optional[str] = None
    thread_id: Optional[str] = None
    user_id: Optional[str] = None
    
    # Processing state (for agents that process files)
    is_processed: bool = False
    processing_result: Optional[Dict[str, Any]] = None
    
    # Extensible metadata
    custom_metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        # Handle Enum vs String for status
        if isinstance(self.status, Enum):
            result['status'] = self.status.value
        else:
            result['status'] = str(self.status)
            
        # Handle Enum vs String for file_type
        if isinstance(self.file_type, Enum):
            result['file_type'] = self.file_type.value
        else:
            result['file_type'] = str(self.file_type)
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentFileMetadata':
        """Create from dictionary"""
        data['status'] = FileStatus(data.get('status', 'active'))
        data['file_type'] = FileType(data.get('file_type', 'other'))
        return cls(**data)
    
    def to_orchestrator_format(self) -> Dict[str, Any]:
        """Convert to orchestrator-compatible format"""
        return {
            "file_id": self.file_id,
            "file_name": self.original_name,
            "file_path": self.storage_path,
            "file_type": self.file_type.value,
            "mime_type": self.mime_type,
            "size": self.size_bytes,
            "agent_id": self.agent_id,
            "thread_id": self.thread_id,
            "orchestrator_content_id": self.orchestrator_content_id
        }


# =============================================================================
# FILE TYPE DETECTION
# =============================================================================

class FileTypeDetector:
    """Detects file type from filename and mime type"""
    
    EXTENSION_MAP = {
        # Spreadsheets
        '.csv': FileType.SPREADSHEET,
        '.xlsx': FileType.SPREADSHEET,
        '.xls': FileType.SPREADSHEET,
        '.ods': FileType.SPREADSHEET,
        
        # Documents
        '.pdf': FileType.DOCUMENT,
        '.doc': FileType.DOCUMENT,
        '.docx': FileType.DOCUMENT,
        '.txt': FileType.DOCUMENT,
        '.md': FileType.DOCUMENT,
        '.rtf': FileType.DOCUMENT,
        
        # Images
        '.jpg': FileType.IMAGE,
        '.jpeg': FileType.IMAGE,
        '.png': FileType.IMAGE,
        '.gif': FileType.IMAGE,
        '.bmp': FileType.IMAGE,
        '.webp': FileType.IMAGE,
        '.svg': FileType.IMAGE,
        
        # Data
        '.json': FileType.DATA,
        '.xml': FileType.DATA,
        '.yaml': FileType.DATA,
        '.yml': FileType.DATA,
        
        # Code
        '.py': FileType.CODE,
        '.js': FileType.CODE,
        '.ts': FileType.CODE,
        '.java': FileType.CODE,
        '.cpp': FileType.CODE,
        '.c': FileType.CODE,
        '.h': FileType.CODE,
        '.go': FileType.CODE,
        '.rs': FileType.CODE,
        
        # Archives
        '.zip': FileType.ARCHIVE,
        '.tar': FileType.ARCHIVE,
        '.gz': FileType.ARCHIVE,
        '.rar': FileType.ARCHIVE,
        '.7z': FileType.ARCHIVE,
    }
    
    MIME_TYPE_MAP = {
        'image/': FileType.IMAGE,
        'text/csv': FileType.SPREADSHEET,
        'application/vnd.ms-excel': FileType.SPREADSHEET,
        'application/vnd.openxmlformats-officedocument.spreadsheetml': FileType.SPREADSHEET,
        'application/pdf': FileType.DOCUMENT,
        'application/json': FileType.DATA,
        'application/xml': FileType.DATA,
        'application/zip': FileType.ARCHIVE,
    }
    
    @classmethod
    def detect(cls, filename: str, mime_type: Optional[str] = None) -> FileType:
        """Detect file type from filename and optional mime type"""
        ext = Path(filename).suffix.lower()
        
        # Check extension first
        if ext in cls.EXTENSION_MAP:
            return cls.EXTENSION_MAP[ext]
        
        # Check mime type
        if mime_type:
            for mime_prefix, file_type in cls.MIME_TYPE_MAP.items():
                if mime_type.startswith(mime_prefix):
                    return file_type
        
        return FileType.OTHER


# =============================================================================
# AGENT FILE MANAGER
# =============================================================================

class AgentFileManager:
    """
    Robust file manager for agents.
    
    Features:
    - Persistent file registry (survives restarts)
    - Automatic file type detection
    - Checksum verification
    - Expiration and cleanup
    - Thread-safe operations
    - Async support
    - Orchestrator integration
    - Extensible hooks
    """
    
    def __init__(
        self,
        agent_id: str,
        storage_dir: str,
        registry_filename: str = "file_registry.json",
        default_ttl_hours: Optional[int] = None,
        auto_cleanup: bool = True,
        cleanup_interval_hours: int = 1
    ):
        """
        Initialize the file manager.
        
        Args:
            agent_id: Unique identifier for this agent
            storage_dir: Directory to store files
            registry_filename: Name of the registry file
            default_ttl_hours: Default time-to-live for files (None = no expiration)
            auto_cleanup: Whether to automatically clean up expired files
            cleanup_interval_hours: How often to run cleanup
        """
        self.agent_id = agent_id
        self.storage_dir = Path(storage_dir).resolve()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.registry_path = self.storage_dir / registry_filename
        self.default_ttl_hours = default_ttl_hours
        self.auto_cleanup = auto_cleanup
        self.cleanup_interval_hours = cleanup_interval_hours
        
        # Thread-safe registry
        self._registry: Dict[str, AgentFileMetadata] = {}
        self._lock = threading.RLock()
        
        # Load existing registry
        self._load_registry()
        
        # Hooks for extensibility
        self._on_file_registered: List[Callable] = []
        self._on_file_accessed: List[Callable] = []
        self._on_file_deleted: List[Callable] = []
        
        # Start cleanup task if enabled
        if auto_cleanup:
            self._start_cleanup_task()
            
        # CMS Integration
        try:
            from backend.services.content_management_service import ContentManagementService
            self.cms = ContentManagementService()
            logger.info("AgentFileManager connected to ContentManagementService")
        except Exception as e:
            logger.warning(f"AgentFileManager failed to connect to CMS: {e}")
            self.cms = None
        
        logger.info(f"AgentFileManager initialized for {agent_id} at {storage_dir}")
    
    # =========================================================================
    # REGISTRY MANAGEMENT
    # =========================================================================
    
    def _load_registry(self):
        """Load file registry from disk"""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._registry = {
                        k: AgentFileMetadata.from_dict(v) 
                        for k, v in data.items()
                    }
                logger.info(f"Loaded {len(self._registry)} files from registry")
                
                # Verify files still exist
                self._verify_registry()
                
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")
                self._registry = {}
    
    def _save_registry(self):
        """Save file registry to disk"""
        try:
            with open(self.registry_path, 'w', encoding='utf-8') as f:
                json.dump(
                    {k: v.to_dict() for k, v in self._registry.items()},
                    f, indent=2
                )
            logger.debug(f"[FILE_MANAGER] Registry saved to {self.registry_path}, {len(self._registry)} entries")
        except Exception as e:
            logger.error(f"Failed to save registry to {self.registry_path}: {e}", exc_info=True)
    
    def _verify_registry(self):
        """Verify all files in registry still exist"""
        missing = []
        for file_id, metadata in self._registry.items():
            if not os.path.exists(metadata.storage_path):
                missing.append(file_id)
                metadata.status = FileStatus.DELETED
        
        if missing:
            logger.warning(f"Found {len(missing)} missing files in registry")
            self._save_registry()
    
    # =========================================================================
    # FILE OPERATIONS
    # =========================================================================
    
    def _generate_file_id(self) -> str:
        """Generate unique file ID"""
        return str(uuid.uuid4())
    
    def _compute_checksum(self, content: bytes) -> str:
        """Compute SHA256 checksum"""
        return hashlib.sha256(content).hexdigest()
    
    def _get_storage_path(self, file_id: str, filename: str) -> Path:
        """Get storage path for a file"""
        ext = Path(filename).suffix
        return self.storage_dir / f"{file_id}{ext}"
    
    async def register_file(
        self,
        content: bytes,
        filename: str,
        mime_type: Optional[str] = None,
        file_type: Optional[FileType] = None,
        ttl_hours: Optional[int] = None,
        orchestrator_content_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> AgentFileMetadata:
        """
        Register a new file.
        
        Args:
            content: File content as bytes
            filename: Original filename
            mime_type: MIME type (auto-detected if not provided)
            file_type: File type (auto-detected if not provided)
            ttl_hours: Time-to-live in hours (uses default if not provided)
            orchestrator_content_id: ID from orchestrator's content system
            thread_id: Associated conversation thread
            user_id: Associated user
            custom_metadata: Additional metadata
            tags: Searchable tags
        
        Returns:
            AgentFileMetadata for the registered file
        """
        with self._lock:
            file_id = self._generate_file_id()
            
            # Detect MIME type
            if not mime_type:
                mime_type, _ = mimetypes.guess_type(filename)
                mime_type = mime_type or 'application/octet-stream'
            
            # Detect file type
            if not file_type:
                file_type = FileTypeDetector.detect(filename, mime_type)
            
            # Get storage path
            storage_path = self._get_storage_path(file_id, filename)
            
            # Write file
            with open(storage_path, 'wb') as f:
                f.write(content)
            
            # Compute checksum
            checksum = self._compute_checksum(content)
            
            # Calculate expiration
            ttl = ttl_hours if ttl_hours is not None else self.default_ttl_hours
            expires_at = None
            if ttl:
                expires_at = (datetime.utcnow() + timedelta(hours=ttl)).isoformat()
            
            # Create metadata
            metadata = AgentFileMetadata(
                file_id=file_id,
                agent_id=self.agent_id,
                original_name=filename,
                storage_path=str(storage_path),
                size_bytes=len(content),
                checksum=checksum,
                mime_type=mime_type,
                file_type=file_type,
                expires_at=expires_at,
                orchestrator_content_id=orchestrator_content_id,
                thread_id=thread_id,
                user_id=user_id,
                custom_metadata=custom_metadata or {},
                tags=tags or []
            )
            
            # Register
            self._registry[file_id] = metadata
            self._save_registry()
            logger.info(f"[FILE_MANAGER_DEBUG] Registry saved. Total entries: {len(self._registry)}, New file_id: {file_id}")
            
            # Register
            self._registry[file_id] = metadata
            self._save_registry()
            logger.info(f"[FILE_MANAGER_DEBUG] Registry saved. Total entries: {len(self._registry)}, New file_id: {file_id}")
            
            # CMS Sync
            if self.cms:
                try:
                    from backend.services.content_management_service import ContentType, ContentSource, ContentPriority
                    # Map metadata to CMS types
                    cms_type = ContentType.OTHER
                    if file_type == FileType.IMAGE: cms_type = ContentType.IMAGE
                    elif file_type == FileType.DOCUMENT: cms_type = ContentType.DOCUMENT
                    elif file_type == FileType.SPREADSHEET: cms_type = ContentType.SPREADSHEET
                    elif file_type == FileType.SCREENSHOT: cms_type = ContentType.SCREENSHOT
                    
                    cms_meta = await self.cms.register_content(
                        content=content,
                        name=filename,
                        source=ContentSource.AGENT_OUTPUT,
                        content_type=cms_type,
                        priority=ContentPriority.MEDIUM,
                        tags=tags + [self.agent_id, "file_manager"],
                        thread_id=thread_id,
                        user_id=user_id,
                        mime_type=mime_type
                    )
                    metadata.orchestrator_content_id = cms_meta.id
                    self._save_registry() # Save again with CMS ID
                    logger.info(f"Synced file {filename} to CMS (ID: {cms_meta.id})")
                except Exception as cms_err:
                    logger.warning(f"Failed to sync file to CMS: {cms_err}")

            # Call hooks
            for hook in self._on_file_registered:
                try:
                    hook(metadata)
                except Exception as e:
                    logger.warning(f"Hook error: {e}")
            
            logger.info(f"Registered file: {filename} -> {file_id} ({file_type.value})")
            return metadata
    
    def get_file(self, file_id: str) -> Optional[AgentFileMetadata]:
        """Get file metadata by ID"""
        with self._lock:
            metadata = self._registry.get(file_id)
            
            if metadata and metadata.status == FileStatus.ACTIVE:
                # Check expiration
                if metadata.expires_at:
                    if datetime.utcnow() > datetime.fromisoformat(metadata.expires_at):
                        metadata.status = FileStatus.EXPIRED
                        self._save_registry()
                        return None
                
                # Update access stats
                metadata.accessed_at = datetime.utcnow().isoformat()
                metadata.access_count += 1
                self._save_registry()
                
                # Call hooks
                for hook in self._on_file_accessed:
                    try:
                        hook(metadata)
                    except Exception as e:
                        logger.warning(f"Hook error: {e}")
                
                return metadata
            
            return None
    
    def get_file_content(self, file_id: str) -> Optional[bytes]:
        """Get file content by ID"""
        metadata = self.get_file(file_id)
        if not metadata:
            return None
        
        try:
            with open(metadata.storage_path, 'rb') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read file {file_id}: {e}")
            return None
    
    def get_file_path(self, file_id: str) -> Optional[str]:
        """Get file path by ID"""
        metadata = self.get_file(file_id)
        return metadata.storage_path if metadata else None
    
    def verify_file(self, file_id: str) -> bool:
        """Verify that a file exists and is valid"""
        metadata = self._registry.get(file_id)
        if not metadata:
            return False
        
        if metadata.status != FileStatus.ACTIVE:
            return False
        
        if not os.path.exists(metadata.storage_path):
            metadata.status = FileStatus.DELETED
            self._save_registry()
            return False
        
        # Check expiration
        if metadata.expires_at:
            if datetime.utcnow() > datetime.fromisoformat(metadata.expires_at):
                metadata.status = FileStatus.EXPIRED
                self._save_registry()
                return False
        
        return True
    
    def delete_file(self, file_id: str) -> bool:
        """Delete a file"""
        with self._lock:
            metadata = self._registry.get(file_id)
            if not metadata:
                return False
            
            # Delete physical file
            try:
                if os.path.exists(metadata.storage_path):
                    os.remove(metadata.storage_path)
            except Exception as e:
                logger.error(f"Failed to delete file {file_id}: {e}")
            
            # Update status
            metadata.status = FileStatus.DELETED
            self._save_registry()
            
            # Call hooks
            for hook in self._on_file_deleted:
                try:
                    hook(metadata)
                except Exception as e:
                    logger.warning(f"Hook error: {e}")
            
            logger.info(f"Deleted file: {file_id}")
            return True
    
    # =========================================================================
    # QUERY OPERATIONS
    # =========================================================================
    
    def list_files(
        self,
        status: Optional[FileStatus] = FileStatus.ACTIVE,
        file_type: Optional[FileType] = None,
        thread_id: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[AgentFileMetadata]:
        """List files with optional filtering"""
        results = []
        
        for metadata in self._registry.values():
            # Filter by status
            if status and metadata.status != status:
                continue
            
            # Filter by file type
            if file_type and metadata.file_type != file_type:
                continue
            
            # Filter by thread
            if thread_id and metadata.thread_id != thread_id:
                continue
            
            # Filter by tags
            if tags and not any(t in metadata.tags for t in tags):
                continue
            
            results.append(metadata)
        
        return results
    
    def get_by_orchestrator_id(self, orchestrator_content_id: str) -> Optional[AgentFileMetadata]:
        """Get file by orchestrator content ID"""
        for metadata in self._registry.values():
            if metadata.orchestrator_content_id == orchestrator_content_id:
                return metadata
        return None
    
    # =========================================================================
    # PROCESSING STATE
    # =========================================================================
    
    def mark_as_processed(
        self,
        file_id: str,
        processing_result: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Mark a file as processed"""
        with self._lock:
            metadata = self._registry.get(file_id)
            if not metadata:
                return False
            
            metadata.is_processed = True
            metadata.processing_result = processing_result
            metadata.updated_at = datetime.utcnow().isoformat()
            self._save_registry()
            
            return True
    
    def get_processing_result(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get processing result for a file"""
        metadata = self._registry.get(file_id)
        if metadata and metadata.is_processed:
            return metadata.processing_result
        return None
    
    # =========================================================================
    # CLEANUP
    # =========================================================================
    
    def cleanup_expired(self) -> int:
        """Remove expired files"""
        now = datetime.utcnow()
        expired = []
        
        for file_id, metadata in self._registry.items():
            if metadata.status != FileStatus.ACTIVE:
                continue
            
            if metadata.expires_at:
                if now > datetime.fromisoformat(metadata.expires_at):
                    expired.append(file_id)
        
        for file_id in expired:
            self.delete_file(file_id)
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired files")
        
        return len(expired)
    
    def cleanup_old(self, max_age_hours: int = 24) -> int:
        """Remove files older than specified hours"""
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        old_files = []
        
        for file_id, metadata in self._registry.items():
            if metadata.status != FileStatus.ACTIVE:
                continue
            
            created_at = datetime.fromisoformat(metadata.created_at)
            if created_at < cutoff:
                old_files.append(file_id)
        
        for file_id in old_files:
            self.delete_file(file_id)
        
        if old_files:
            logger.info(f"Cleaned up {len(old_files)} old files")
        
        return len(old_files)
    
    def _start_cleanup_task(self):
        """Start background cleanup task"""
        def cleanup_loop():
            while True:
                try:
                    import time
                    time.sleep(self.cleanup_interval_hours * 3600)
                    self.cleanup_expired()
                except Exception as e:
                    logger.error(f"Cleanup error: {e}")
        
        thread = threading.Thread(target=cleanup_loop, daemon=True)
        thread.start()
    
    # =========================================================================
    # HOOKS
    # =========================================================================
    
    def on_file_registered(self, callback: Callable[[AgentFileMetadata], None]):
        """Register a callback for when a file is registered"""
        self._on_file_registered.append(callback)
    
    def on_file_accessed(self, callback: Callable[[AgentFileMetadata], None]):
        """Register a callback for when a file is accessed"""
        self._on_file_accessed.append(callback)
    
    def on_file_deleted(self, callback: Callable[[AgentFileMetadata], None]):
        """Register a callback for when a file is deleted"""
        self._on_file_deleted.append(callback)
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get file manager statistics"""
        active_files = [m for m in self._registry.values() if m.status == FileStatus.ACTIVE]
        
        total_size = sum(m.size_bytes for m in active_files)
        
        by_type = {}
        for file_type in FileType:
            type_files = [m for m in active_files if m.file_type == file_type]
            if type_files:
                by_type[file_type.value] = {
                    "count": len(type_files),
                    "size_bytes": sum(m.size_bytes for m in type_files)
                }
        
        return {
            "agent_id": self.agent_id,
            "total_files": len(active_files),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "by_type": by_type,
            "storage_dir": str(self.storage_dir)
        }
