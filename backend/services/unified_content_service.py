"""
Unified Content Management Service

This service consolidates file management and artifact management into a single,
standardized system for handling all content in the orchestrator.
"""

import os
import uuid
import json
import gzip
import hashlib
import logging
import mimetypes
import threading
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger("UnifiedContentService")

# Storage directories
STORAGE_BASE = Path("storage")
CONTENT_DIR = STORAGE_BASE / "content"
USER_UPLOADS_DIR = CONTENT_DIR / "uploads"
AGENT_FILES_DIR = CONTENT_DIR / "agent_files"
ARTIFACTS_DIR = CONTENT_DIR / "artifacts"
TEMP_DIR = CONTENT_DIR / "temp"

for dir_path in [USER_UPLOADS_DIR, AGENT_FILES_DIR, ARTIFACTS_DIR, TEMP_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


class ContentType(str, Enum):
    IMAGE = "image"
    DOCUMENT = "document"
    SPREADSHEET = "spreadsheet"
    CODE = "code"
    DATA = "data"
    ARCHIVE = "archive"
    CANVAS = "canvas"
    SCREENSHOT = "screenshot"
    RESULT = "result"
    PLAN = "plan"
    CONVERSATION = "conversation"
    SUMMARY = "summary"
    OTHER = "other"


class ContentSource(str, Enum):
    USER_UPLOAD = "user_upload"
    AGENT_OUTPUT = "agent_output"
    SYSTEM_GENERATED = "system_generated"
    EMAIL_ATTACHMENT = "email_attachment"
    BROWSER_CAPTURE = "browser_capture"


class ContentPriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    EPHEMERAL = "ephemeral"


class RetentionPolicy(str, Enum):
    PERMANENT = "permanent"
    SESSION = "session"
    TTL = "ttl"
    ON_DEMAND = "on_demand"


CONTENT_TYPE_DIRS = {
    ContentType.IMAGE: CONTENT_DIR / "images",
    ContentType.DOCUMENT: CONTENT_DIR / "documents",
    ContentType.SPREADSHEET: CONTENT_DIR / "spreadsheets",
    ContentType.CODE: CONTENT_DIR / "code",
    ContentType.DATA: CONTENT_DIR / "data",
    ContentType.CANVAS: ARTIFACTS_DIR / "canvas",
    ContentType.SCREENSHOT: ARTIFACTS_DIR / "screenshots",
    ContentType.RESULT: ARTIFACTS_DIR / "results",
    ContentType.PLAN: ARTIFACTS_DIR / "plans",
    ContentType.CONVERSATION: ARTIFACTS_DIR / "conversations",
    ContentType.SUMMARY: ARTIFACTS_DIR / "summaries",
    ContentType.OTHER: CONTENT_DIR / "other",
}

for dir_path in CONTENT_TYPE_DIRS.values():
    dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class AgentContentMapping:
    content_id: str
    agent_id: str
    agent_content_id: str
    agent_endpoint: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    verified_at: Optional[str] = None
    is_valid: bool = True


@dataclass
class UnifiedContentMetadata:
    id: str
    name: str
    content_type: ContentType
    source: ContentSource
    storage_path: str
    size_bytes: int
    checksum: str
    is_compressed: bool = False
    mime_type: str = "application/octet-stream"
    user_id: str = "system"
    thread_id: Optional[str] = None
    agent_mappings: Dict[str, AgentContentMapping] = field(default_factory=dict)
    priority: ContentPriority = ContentPriority.MEDIUM
    retention_policy: RetentionPolicy = RetentionPolicy.TTL
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    accessed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    expires_at: Optional[str] = None
    access_count: int = 0
    summary: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    is_artifact: bool = False
    original_size: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['content_type'] = self.content_type.value
        result['source'] = self.source.value
        result['priority'] = self.priority.value
        result['retention_policy'] = self.retention_policy.value
        result['agent_mappings'] = {k: asdict(v) for k, v in self.agent_mappings.items()}
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnifiedContentMetadata':
        data['content_type'] = ContentType(data['content_type'])
        data['source'] = ContentSource(data['source'])
        data['priority'] = ContentPriority(data['priority'])
        data['retention_policy'] = RetentionPolicy(data['retention_policy'])
        data['agent_mappings'] = {k: AgentContentMapping(**v) for k, v in data.get('agent_mappings', {}).items()}
        return cls(**data)
    
    def to_file_object(self) -> Dict[str, Any]:
        return {
            "file_id": self.id,
            "file_name": self.name,
            "file_path": self.storage_path,
            "file_type": self.content_type.value,
            "mime_type": self.mime_type,
            "size": self.size_bytes,
            "source": self.source.value,
            "thread_id": self.thread_id,
        }
    
    def to_reference(self) -> 'ContentReference':
        return ContentReference(
            id=self.id,
            name=self.name,
            content_type=self.content_type,
            summary=self.summary or f"{self.content_type.value}: {self.name}",
            size_bytes=self.size_bytes
        )


    def to_reference(self) -> 'ContentReference':
        """Convert to lightweight reference for context inclusion"""
        return ContentReference(
            id=self.id,
            name=self.name,
            content_type=self.content_type,
            summary=self.summary or f"{self.content_type.value}: {self.name}",
            size_bytes=self.size_bytes
        )


@dataclass
class ContentReference:
    """Lightweight reference to content for context inclusion"""
    id: str
    name: str
    content_type: ContentType
    summary: str
    size_bytes: int
    
    def to_context_string(self) -> str:
        """Generate a context-friendly string representation"""
        size_kb = self.size_bytes / 1024
        return f"[CONTENT:{self.id}] {self.name} ({self.content_type.value}, {size_kb:.1f}KB)\n  Summary: {self.summary}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "content_type": self.content_type.value,
            "summary": self.summary,
            "size_bytes": self.size_bytes
        }


# =============================================================================
# NOTE: AgentFileInterface moved to agents/standard_file_interface.py
# =============================================================================



# =============================================================================
# UNIFIED CONTENT SERVICE
# =============================================================================

class UnifiedContentService:
    """
    Central service for managing all content across the orchestrator and agents.
    
    This consolidates:
    - File management (user uploads, agent files)
    - Artifact management (large content, canvas, screenshots)
    - Agent content synchronization
    - Lifecycle management
    """
    
    # Compression threshold (bytes)
    COMPRESSION_THRESHOLD = 1024
    
    # Size thresholds for automatic artifact creation
    ARTIFACT_THRESHOLDS = {
        ContentType.CANVAS: 500,
        ContentType.SCREENSHOT: 100,
        ContentType.RESULT: 2000,
        ContentType.CONVERSATION: 5000,
    }
    
    def __init__(self, storage_dir: str = "storage/content"):
        self._registry: Dict[str, UnifiedContentMetadata] = {}
        self._registry_path = Path(storage_dir) / "content_registry.json"
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._load_registry()
        logger.info(f"UnifiedContentService initialized with {len(self._registry)} items")
    
    def _load_registry(self):
        """Load content registry from disk"""
        if self._registry_path.exists():
            try:
                with open(self._registry_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._registry = {
                        k: UnifiedContentMetadata.from_dict(v) for k, v in data.items()
                    }
                logger.info(f"Loaded {len(self._registry)} content items from registry")
            except Exception as e:
                logger.error(f"Failed to load content registry: {e}")
                self._registry = {}
    
    def _save_registry(self):
        """Persist content registry to disk"""
        try:
            with open(self._registry_path, 'w', encoding='utf-8') as f:
                json.dump(
                    {k: v.to_dict() for k, v in self._registry.items()},
                    f, indent=2
                )
        except Exception as e:
            logger.error(f"Failed to save content registry: {e}")
    
    def _compute_checksum(self, content: bytes) -> str:
        """Compute SHA256 checksum"""
        return hashlib.sha256(content).hexdigest()
    
    def _generate_id(self) -> str:
        """Generate unique content ID"""
        return str(uuid.uuid4())
    
    def _determine_content_type(self, filename: str, mime_type: str) -> ContentType:
        """Determine content type based on filename and mime type"""
        ext = Path(filename).suffix.lower()
        
        # Spreadsheets
        if ext in ['.csv', '.xlsx', '.xls', '.ods']:
            return ContentType.SPREADSHEET
        
        # Images
        if mime_type and mime_type.startswith('image/'):
            return ContentType.IMAGE
        
        # Documents
        if ext in ['.pdf', '.doc', '.docx', '.txt', '.md', '.rtf']:
            return ContentType.DOCUMENT
        
        # Code
        if ext in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.go', '.rs']:
            return ContentType.CODE
        
        # Data
        if ext in ['.json', '.xml', '.yaml', '.yml']:
            return ContentType.DATA
        
        # Archives
        if ext in ['.zip', '.tar', '.gz', '.rar', '.7z']:
            return ContentType.ARCHIVE
        
        return ContentType.OTHER
    
    def _get_storage_path(self, content_id: str, content_type: ContentType, ext: str) -> Path:
        """Get storage path for content"""
        base_dir = CONTENT_TYPE_DIRS.get(content_type, CONTENT_DIR / "other")
        return base_dir / f"{content_id}{ext}"
    
    def _calculate_expiration(self, priority: ContentPriority, ttl_hours: Optional[int] = None) -> Optional[str]:
        """Calculate expiration time based on priority"""
        if ttl_hours:
            return (datetime.utcnow() + timedelta(hours=ttl_hours)).isoformat()
        
        ttl_map = {
            ContentPriority.CRITICAL: None,  # Never expires
            ContentPriority.HIGH: 30 * 24,   # 30 days
            ContentPriority.MEDIUM: 7 * 24,  # 7 days
            ContentPriority.LOW: 24,         # 24 hours
            ContentPriority.EPHEMERAL: 1,    # 1 hour
        }
        
        hours = ttl_map.get(priority)
        if hours:
            return (datetime.utcnow() + timedelta(hours=hours)).isoformat()
        return None
    
    def _generate_summary(self, content: Any, content_type: ContentType) -> str:
        """Generate a brief summary of the content"""
        if isinstance(content, bytes):
            return f"Binary content ({len(content)} bytes)"
        
        if isinstance(content, str):
            preview = content[:200].replace('\n', ' ')
            if len(content) > 200:
                preview += "..."
            return preview
        
        if isinstance(content, dict):
            keys = list(content.keys())[:5]
            key_str = ", ".join(str(k) for k in keys)
            if len(content) > 5:
                key_str += f", ... (+{len(content) - 5} more)"
            return f"Object with keys: {key_str}"
        
        if isinstance(content, list):
            if len(content) == 0:
                return "Empty list"
            sample = str(content[0])[:50]
            return f"List of {len(content)} items. First: {sample}..."
        
        return f"{content_type.value} content"
    
    # =========================================================================
    # CONTENT REGISTRATION
    # =========================================================================
    
    async def register_content(
        self,
        content: Union[bytes, str, Dict, List],
        name: str,
        source: ContentSource,
        user_id: str = "system",
        thread_id: Optional[str] = None,
        content_type: Optional[ContentType] = None,
        mime_type: Optional[str] = None,
        priority: ContentPriority = ContentPriority.MEDIUM,
        retention_policy: RetentionPolicy = RetentionPolicy.TTL,
        ttl_hours: Optional[int] = None,
        tags: Optional[List[str]] = None,
        is_artifact: bool = False
    ) -> UnifiedContentMetadata:
        """
        Register any content (file or artifact) in the unified system.
        
        Args:
            content: The content to store (bytes for files, any for artifacts)
            name: Human-readable name
            source: Where the content came from
            user_id: Owner user ID
            thread_id: Associated conversation thread
            content_type: Type of content (auto-detected if not provided)
            mime_type: MIME type (auto-detected if not provided)
            priority: Retention priority
            retention_policy: How to handle retention
            ttl_hours: Custom TTL in hours
            tags: Searchable tags
            is_artifact: Whether this is an artifact (large content)
        
        Returns:
            UnifiedContentMetadata for the registered content
        """
        with self._lock:
            content_id = self._generate_id()
            
            # Determine MIME type
            if not mime_type:
                mime_type, _ = mimetypes.guess_type(name)
                mime_type = mime_type or 'application/octet-stream'
            
            # Determine content type
            if not content_type:
                content_type = self._determine_content_type(name, mime_type)
            
            # Convert content to bytes for storage
            if isinstance(content, bytes):
                content_bytes = content
            elif isinstance(content, str):
                content_bytes = content.encode('utf-8')
            else:
                content_bytes = json.dumps(content, default=str, ensure_ascii=False).encode('utf-8')
            
            original_size = len(content_bytes)
            
            # Compress if large
            is_compressed = original_size > self.COMPRESSION_THRESHOLD
            if is_compressed:
                content_bytes = gzip.compress(content_bytes)
            
            # Determine storage path
            ext = Path(name).suffix or '.bin'
            if is_compressed:
                ext += '.gz'
            storage_path = self._get_storage_path(content_id, content_type, ext)
            
            # Write to disk
            with open(storage_path, 'wb') as f:
                f.write(content_bytes)
            
            # Compute checksum
            checksum = self._compute_checksum(content_bytes)
            
            # Generate summary
            summary = self._generate_summary(content, content_type)
            
            # Calculate expiration
            expires_at = self._calculate_expiration(priority, ttl_hours)
            
            # Create metadata
            metadata = UnifiedContentMetadata(
                id=content_id,
                name=name,
                content_type=content_type,
                source=source,
                storage_path=str(storage_path),
                size_bytes=len(content_bytes),
                checksum=checksum,
                is_compressed=is_compressed,
                mime_type=mime_type,
                user_id=user_id,
                thread_id=thread_id,
                priority=priority,
                retention_policy=retention_policy,
                expires_at=expires_at,
                summary=summary,
                tags=tags or [],
                is_artifact=is_artifact,
                original_size=original_size
            )
            
            # Register
            self._registry[content_id] = metadata
            self._save_registry()
            
            logger.info(f"Registered content: {name} -> {content_id} ({content_type.value}, {metadata.size_bytes} bytes)")
            return metadata
    
    async def register_user_upload(
        self,
        file_content: bytes,
        filename: str,
        user_id: str,
        thread_id: Optional[str] = None,
        mime_type: Optional[str] = None
    ) -> UnifiedContentMetadata:
        """Convenience method for registering user uploads"""
        return await self.register_content(
            content=file_content,
            name=filename,
            source=ContentSource.USER_UPLOAD,
            user_id=user_id,
            thread_id=thread_id,
            mime_type=mime_type,
            priority=ContentPriority.HIGH
        )
    
    async def register_agent_output(
        self,
        content: Union[bytes, str, Dict],
        name: str,
        agent_id: str,
        user_id: str,
        thread_id: Optional[str] = None,
        content_type: Optional[ContentType] = None,
        is_temporary: bool = False
    ) -> UnifiedContentMetadata:
        """Convenience method for registering agent outputs"""
        priority = ContentPriority.EPHEMERAL if is_temporary else ContentPriority.MEDIUM
        
        metadata = await self.register_content(
            content=content,
            name=name,
            source=ContentSource.AGENT_OUTPUT,
            user_id=user_id,
            thread_id=thread_id,
            content_type=content_type,
            priority=priority,
            tags=[f"agent:{agent_id}"]
        )
        return metadata
    
    async def register_artifact(
        self,
        content: Any,
        name: str,
        content_type: ContentType,
        thread_id: str,
        description: Optional[str] = None,
        priority: ContentPriority = ContentPriority.MEDIUM,
        ttl_hours: Optional[int] = None
    ) -> UnifiedContentMetadata:
        """Convenience method for registering artifacts (large content)"""
        metadata = await self.register_content(
            content=content,
            name=name,
            source=ContentSource.SYSTEM_GENERATED,
            thread_id=thread_id,
            content_type=content_type,
            priority=priority,
            ttl_hours=ttl_hours,
            is_artifact=True
        )
        
        if description:
            metadata.summary = description
            self._save_registry()
        
        return metadata

    
    # =========================================================================
    # CONTENT RETRIEVAL
    # =========================================================================
    
    def get_content(self, content_id: str, update_access: bool = True) -> Optional[Tuple[UnifiedContentMetadata, Any]]:
        """
        Retrieve content by ID.
        
        Returns:
            Tuple of (metadata, content) or None if not found
        """
        with self._lock:
            if content_id not in self._registry:
                logger.warning(f"Content not found: {content_id}")
                return None
            
            metadata = self._registry[content_id]
            
            # Check expiration
            if metadata.expires_at:
                if datetime.utcnow() > datetime.fromisoformat(metadata.expires_at):
                    logger.info(f"Content expired: {content_id}")
                    self.delete_content(content_id)
                    return None
            
            # Read content
            try:
                with open(metadata.storage_path, 'rb') as f:
                    content_bytes = f.read()
                
                # Decompress if needed
                if metadata.is_compressed:
                    content_bytes = gzip.decompress(content_bytes)
                
                # Parse content based on type
                if metadata.is_artifact or metadata.content_type in [
                    ContentType.RESULT, ContentType.PLAN, ContentType.DATA
                ]:
                    try:
                        content = json.loads(content_bytes.decode('utf-8'))
                    except json.JSONDecodeError:
                        content = content_bytes.decode('utf-8')
                elif metadata.content_type in [ContentType.IMAGE, ContentType.ARCHIVE]:
                    content = content_bytes
                else:
                    try:
                        content = content_bytes.decode('utf-8')
                    except UnicodeDecodeError:
                        content = content_bytes
                
                # Update access stats
                if update_access:
                    metadata.accessed_at = datetime.utcnow().isoformat()
                    metadata.access_count += 1
                    self._save_registry()
                
                return metadata, content
                
            except Exception as e:
                logger.error(f"Failed to retrieve content {content_id}: {e}")
                return None
    
    def get_metadata(self, content_id: str) -> Optional[UnifiedContentMetadata]:
        """Get metadata only (no content loading)"""
        return self._registry.get(content_id)
    
    def get_content_bytes(self, content_id: str) -> Optional[bytes]:
        """Get raw content bytes"""
        result = self.get_content(content_id)
        if not result:
            return None
        
        metadata, content = result
        if isinstance(content, bytes):
            return content
        elif isinstance(content, str):
            return content.encode('utf-8')
        else:
            return json.dumps(content, default=str).encode('utf-8')
    
    def get_content_path(self, content_id: str) -> Optional[str]:
        """Get storage path for content"""
        metadata = self._registry.get(content_id)
        return metadata.storage_path if metadata else None
    
    # =========================================================================
    # CONTENT QUERIES
    # =========================================================================
    
    def get_by_thread(self, thread_id: str) -> List[UnifiedContentMetadata]:
        """Get all content for a thread"""
        return [m for m in self._registry.values() if m.thread_id == thread_id]
    
    def get_by_user(self, user_id: str) -> List[UnifiedContentMetadata]:
        """Get all content for a user"""
        return [m for m in self._registry.values() if m.user_id == user_id]
    
    def get_by_type(self, content_type: ContentType) -> List[UnifiedContentMetadata]:
        """Get all content of a specific type"""
        return [m for m in self._registry.values() if m.content_type == content_type]
    
    def get_by_source(self, source: ContentSource) -> List[UnifiedContentMetadata]:
        """Get all content from a specific source"""
        return [m for m in self._registry.values() if m.source == source]
    
    def get_by_tags(self, tags: List[str]) -> List[UnifiedContentMetadata]:
        """Get content matching any of the tags"""
        return [m for m in self._registry.values() if any(t in m.tags for t in tags)]
    
    def get_files_for_orchestrator(self, thread_id: str) -> List[Dict[str, Any]]:
        """Get files in orchestrator-compatible format"""
        content_list = self.get_by_thread(thread_id)
        return [m.to_file_object() for m in content_list if not m.is_artifact]
    
    # =========================================================================
    # CONTENT DELETION
    # =========================================================================
    
    def delete_content(self, content_id: str) -> bool:
        """Delete content by ID"""
        with self._lock:
            if content_id not in self._registry:
                return False
            
            metadata = self._registry[content_id]
            
            # Delete physical file
            try:
                if os.path.exists(metadata.storage_path):
                    os.remove(metadata.storage_path)
            except Exception as e:
                logger.error(f"Failed to delete file {content_id}: {e}")
            
            # Remove from registry
            del self._registry[content_id]
            self._save_registry()
            
            logger.info(f"Deleted content: {content_id}")
            return True
    
    def cleanup_expired(self) -> int:
        """Remove expired content"""
        now = datetime.utcnow()
        expired_ids = []
        
        for content_id, metadata in self._registry.items():
            if metadata.expires_at:
                expires_at = datetime.fromisoformat(metadata.expires_at)
                if now > expires_at:
                    expired_ids.append(content_id)
        
        for content_id in expired_ids:
            self.delete_content(content_id)
        
        if expired_ids:
            logger.info(f"Cleaned up {len(expired_ids)} expired content items")
        
        return len(expired_ids)
    
    def cleanup_session(self, thread_id: str) -> int:
        """Remove session-only content for a thread"""
        session_content = [
            m.id for m in self._registry.values()
            if m.thread_id == thread_id and m.retention_policy == RetentionPolicy.SESSION
        ]
        
        for content_id in session_content:
            self.delete_content(content_id)
        
        return len(session_content)
    
    # =========================================================================
    # AGENT CONTENT MAPPING
    # =========================================================================
    
    async def upload_to_agent(
        self,
        content_id: str,
        agent_id: str,
        agent_base_url: str,
        upload_endpoint: str = "/upload",
        file_param_name: str = "file",
        force_reupload: bool = False
    ) -> Optional[str]:
        """
        Upload content to an agent and track the mapping.
        
        Returns:
            Agent's content ID if successful, None otherwise
        """
        import httpx
        
        metadata = self._registry.get(content_id)
        if not metadata:
            logger.error(f"Content not found: {content_id}")
            return None
        
        # Check existing mapping
        if agent_id in metadata.agent_mappings and not force_reupload:
            mapping = metadata.agent_mappings[agent_id]
            if mapping.is_valid:
                # Verify the content still exists on agent
                is_valid = await self._verify_agent_content(
                    agent_base_url, mapping.agent_content_id
                )
                if is_valid:
                    logger.info(f"Using existing mapping: {content_id} -> {mapping.agent_content_id}")
                    return mapping.agent_content_id
                else:
                    mapping.is_valid = False
        
        # Read content
        content_bytes = self.get_content_bytes(content_id)
        if not content_bytes:
            logger.error(f"[UPLOAD_DEBUG] Failed to get content bytes for {content_id}")
            return None
        
        # Upload to agent
        upload_url = f"{agent_base_url.rstrip('/')}{upload_endpoint}"
        
        logger.info(f"[UPLOAD_DEBUG] Uploading to {upload_url}")
        logger.info(f"[UPLOAD_DEBUG] File name: {metadata.name}, mime_type: {metadata.mime_type}")
        logger.info(f"[UPLOAD_DEBUG] Content bytes length: {len(content_bytes)}, first 100 bytes: {content_bytes[:100]}")
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                files = {file_param_name: (metadata.name, content_bytes, metadata.mime_type)}
                response = await client.post(upload_url, files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Extract agent's content ID
                    agent_content_id = self._extract_agent_content_id(result)
                    
                    if agent_content_id:
                        # Create mapping
                        mapping = AgentContentMapping(
                            content_id=content_id,
                            agent_id=agent_id,
                            agent_content_id=agent_content_id,
                            agent_endpoint=upload_endpoint,
                            verified_at=datetime.utcnow().isoformat()
                        )
                        metadata.agent_mappings[agent_id] = mapping
                        self._save_registry()
                        
                        logger.info(f"Uploaded content {content_id} to agent {agent_id}: {agent_content_id}")
                        return agent_content_id
                    else:
                        logger.warning(f"Agent {agent_id} returned success but no content ID")
                        return None
                else:
                    # Include response text for debugging
                    response_text = response.text[:500] if response.text else "No response body"
                    logger.error(f"Failed to upload to agent {agent_id}: status={response.status_code}, response={response_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error uploading to agent {agent_id}: {e}", exc_info=True)
            return None
    
    def _extract_agent_content_id(self, response: Dict[str, Any]) -> Optional[str]:
        """Extract content ID from agent response"""
        # Try common patterns
        if isinstance(response, dict):
            if 'result' in response and isinstance(response['result'], dict):
                return response['result'].get('file_id')
            if 'file_id' in response:
                return response['file_id']
            if 'id' in response:
                return response['id']
            if 'content_id' in response:
                return response['content_id']
        return None
    
    async def _verify_agent_content(self, agent_base_url: str, agent_content_id: str) -> bool:
        """Verify content still exists on agent"""
        import httpx
        
        try:
            verify_url = f"{agent_base_url.rstrip('/')}/get_summary"
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(verify_url, data={"file_id": agent_content_id})
                return response.status_code == 200
        except Exception:
            return False
    
    def get_agent_content_id(self, content_id: str, agent_id: str) -> Optional[str]:
        """Get the agent-specific content ID"""
        metadata = self._registry.get(content_id)
        if not metadata:
            return None
        
        mapping = metadata.agent_mappings.get(agent_id)
        return mapping.agent_content_id if mapping and mapping.is_valid else None
    
    # =========================================================================
    # CONTEXT OPTIMIZATION
    # =========================================================================
    
    def should_create_artifact(self, content: Any, content_type: ContentType) -> bool:
        """Determine if content should be stored as an artifact"""
        threshold = self.ARTIFACT_THRESHOLDS.get(content_type, 2000)
        
        if isinstance(content, str):
            return len(content) > threshold
        elif isinstance(content, (dict, list)):
            return len(json.dumps(content, default=str)) > threshold
        elif isinstance(content, bytes):
            return len(content) > threshold
        return False
    
    def compress_for_context(
        self,
        content: Any,
        name: str,
        thread_id: str,
        content_type: ContentType = ContentType.DATA
    ) -> Union[Any, ContentReference]:
        """
        Compress content for context inclusion.
        Returns original content if small, or ContentReference if stored as artifact.
        """
        if not self.should_create_artifact(content, content_type):
            return content
        
        # Store as artifact (synchronous wrapper)
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        metadata = loop.run_until_complete(
            self.register_artifact(
                content=content,
                name=name,
                content_type=content_type,
                thread_id=thread_id
            )
        )
        
        return metadata.to_reference()
    
    def get_optimized_context(
        self,
        thread_id: str,
        max_tokens: int = 8000,
        include_summaries: bool = True
    ) -> Dict[str, Any]:
        """
        Get optimized context for a thread.
        
        Returns:
            {
                "references": List of ContentReference,
                "context_string": Human-readable context,
                "tokens_saved": Estimated tokens saved
            }
        """
        content_list = self.get_by_thread(thread_id)
        
        references = []
        context_lines = []
        total_original_size = 0
        total_reference_size = 0
        
        for metadata in content_list:
            ref = metadata.to_reference()
            references.append(ref)
            
            if include_summaries:
                context_lines.append(ref.to_context_string())
            
            total_original_size += metadata.original_size or metadata.size_bytes
            total_reference_size += len(ref.to_context_string())
        
        context_string = "\n".join(context_lines) if context_lines else ""
        tokens_saved = (total_original_size - total_reference_size) // 4  # ~4 chars per token
        
        return {
            "references": [r.to_dict() for r in references],
            "context_string": context_string,
            "tokens_saved": max(0, tokens_saved),
            "content_count": len(references)
        }
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        total_size = sum(m.size_bytes for m in self._registry.values())
        
        by_type = {}
        for content_type in ContentType:
            type_items = [m for m in self._registry.values() if m.content_type == content_type]
            if type_items:
                by_type[content_type.value] = {
                    "count": len(type_items),
                    "size_bytes": sum(m.size_bytes for m in type_items)
                }
        
        by_source = {}
        for source in ContentSource:
            source_items = [m for m in self._registry.values() if m.source == source]
            if source_items:
                by_source[source.value] = {
                    "count": len(source_items),
                    "size_bytes": sum(m.size_bytes for m in source_items)
                }
        
        return {
            "total_items": len(self._registry),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "by_type": by_type,
            "by_source": by_source
        }


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_content_service: Optional[UnifiedContentService] = None


def get_content_service() -> UnifiedContentService:
    """Get the global content service instance"""
    global _content_service
    if _content_service is None:
        _content_service = UnifiedContentService()
    return _content_service
