"""
Content Management Service

This service consolidates file management, artifact management, and content processing
into a single, standardized system.

Features:
- Unified file/artifact registry with compression and deduplication
- Agent content mapping for cross-agent file sharing
- Lifecycle management with TTL-based expiration
- Context optimization for LLM prompts
"""

import os
import uuid
import json
import gzip
import hashlib
import logging
import mimetypes
import threading
import asyncio
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import io

try:
    import pandas as pd
except ImportError:
    pd = None

import sys
orbimesh_root = Path(__file__).parent.parent.parent.resolve()
if str(orbimesh_root) not in sys.path:
    sys.path.insert(0, str(orbimesh_root))

from backend.services.inference_service import inference_service, InferencePriority
from langchain_core.messages import HumanMessage

logger = logging.getLogger("ContentManagementService")

# Storage directories (centralized)
from backend.storage_config import (
    STORAGE_ROOT as STORAGE_BASE, SYSTEM_DIR, CONTENT_DIR,
    TEMP_DIR as _TEMP_DIR, PROJECT_ROOT,
)

USER_UPLOADS_DIR = CONTENT_DIR / "uploads"
AGENT_FILES_DIR = CONTENT_DIR / "agent_files"
ARTIFACTS_DIR = CONTENT_DIR / "artifacts"
TEMP_DIR = _TEMP_DIR

# Ensure CMS-specific subdirectories exist
for dir_path in [USER_UPLOADS_DIR, AGENT_FILES_DIR, ARTIFACTS_DIR]:
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
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None).isoformat())
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
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None).isoformat())
    accessed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None).isoformat())
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


class ContentManagementService:
    """
    Central service for managing all content across the orchestrator and agents.

    Features:
    - Unified file/artifact registry with compression and deduplication
    - Agent content mapping for cross-agent file sharing
    - Lifecycle management with TTL-based expiration
    - Context optimization for LLM prompts
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

    # Chars per token estimate for context trimming
    CHARS_PER_TOKEN_EST = 4

    def __init__(self, storage_dir: str = None):
        self._registry: Dict[str, UnifiedContentMetadata] = {}
        if storage_dir is None:
            storage_dir = str(CONTENT_DIR)
        self._registry_path = Path(storage_dir) / "content_registry.json"
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._load_registry()
        self._llm_client = None

        logger.info(f"ContentManagementService initialized with {len(self._registry)} items")

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
        if ext in ['.csv', '.xlsx', '.xls', '.ods']:
            return ContentType.SPREADSHEET
        if mime_type and mime_type.startswith('image/'):
            return ContentType.IMAGE
        if ext in ['.pdf', '.doc', '.docx', '.txt', '.md', '.rtf']:
            return ContentType.DOCUMENT
        if ext in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.go', '.rs']:
            return ContentType.CODE
        if ext in ['.json', '.xml', '.yaml', '.yml']:
            return ContentType.DATA
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
            return (datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=ttl_hours)).isoformat()

        ttl_map = {
            ContentPriority.CRITICAL: None,
            ContentPriority.HIGH: 30 * 24,
            ContentPriority.MEDIUM: 7 * 24,
            ContentPriority.LOW: 24,
            ContentPriority.EPHEMERAL: 1,
        }
        hours = ttl_map.get(priority)
        if hours:
            return (datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=hours)).isoformat()
        return None

    def _generate_summary(self, content: Any, content_type: ContentType) -> str:
        """Generate a brief summary of the content (simple preview)"""
        if isinstance(content, bytes):
            return f"Binary content ({len(content)} bytes)"
        if isinstance(content, str):
            preview = content[:200].replace('\n', ' ')
            if len(content) > 200:
                preview += "..."
            return preview
        if isinstance(content, dict):
            return f"Object with keys: {list(content.keys())[:5]}..."
        if isinstance(content, list):
            return f"List of {len(content)} items..."
        return f"{content_type.value} content"

    # =========================================================================
    # CONTENT REGISTRATION & RETRIEVAL
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
        is_artifact: bool = False,
        summary: Optional[str] = None
    ) -> UnifiedContentMetadata:
        """Register content with compression and deduplication."""
        with self._lock:
            # Prepare content bytes
            if isinstance(content, bytes):
                content_bytes = content
            elif isinstance(content, str):
                content_bytes = content.encode('utf-8')
            else:
                content_bytes = json.dumps(content, default=str, ensure_ascii=False).encode('utf-8')

            # Compress if needed
            original_size = len(content_bytes)
            is_compressed = original_size > self.COMPRESSION_THRESHOLD
            final_bytes = gzip.compress(content_bytes) if is_compressed else content_bytes

            # Compute checksum of final bytes
            final_checksum = self._compute_checksum(final_bytes)

            # Dedupe check
            if not is_artifact:
                for meta in self._registry.values():
                    if (meta.checksum == final_checksum and
                        meta.size_bytes == len(final_bytes) and
                        meta.name == name):
                        logger.info(f"♻️ Content deduplicated: {name} -> Existing ID {meta.id}")
                        if tags:
                            meta.tags = list(set(meta.tags) | set(tags))
                        meta.accessed_at = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
                        self._save_registry()
                        return meta

            # New content creation
            content_id = self._generate_id()
            if not mime_type:
                mime_type, _ = mimetypes.guess_type(name)
                mime_type = mime_type or 'application/octet-stream'
            if not content_type:
                content_type = self._determine_content_type(name, mime_type)

            ext = Path(name).suffix or '.bin'
            if is_compressed:
                ext += '.gz'
            storage_path = self._get_storage_path(content_id, content_type, ext)

            with open(storage_path, 'wb') as f:
                f.write(final_bytes)

            if not summary:
                summary = self._generate_summary(content, content_type)
            expires_at = self._calculate_expiration(priority, ttl_hours)

            metadata = UnifiedContentMetadata(
                id=content_id, name=name, content_type=content_type, source=source,
                storage_path=str(storage_path), size_bytes=len(final_bytes), checksum=final_checksum,
                is_compressed=is_compressed, mime_type=mime_type, user_id=user_id,
                thread_id=thread_id, priority=priority, retention_policy=retention_policy,
                expires_at=expires_at, summary=summary, tags=tags or [],
                is_artifact=is_artifact, original_size=original_size
            )

            self._registry[content_id] = metadata
            self._save_registry()
            logger.info(f"Registered content: {name} -> {content_id}")
            return metadata

    def get_content(self, content_id: str, update_access: bool = True) -> Optional[Tuple[UnifiedContentMetadata, Any]]:
        """Retrieve content by ID, decompressing if needed."""
        with self._lock:
            if content_id not in self._registry:
                return None
            metadata = self._registry[content_id]
            
            # Check expiration
            if metadata.expires_at and datetime.now(timezone.utc).replace(tzinfo=None) > datetime.fromisoformat(metadata.expires_at):
                self.delete_content(content_id)
                return None

            try:
                with open(metadata.storage_path, 'rb') as f:
                    content_bytes = f.read()
                if metadata.is_compressed:
                    content_bytes = gzip.decompress(content_bytes)

                # Parse based on content type
                if metadata.is_artifact or metadata.content_type in [ContentType.RESULT, ContentType.PLAN, ContentType.DATA]:
                    try:
                        content = json.loads(content_bytes.decode('utf-8'))
                    except:
                        content = content_bytes.decode('utf-8')
                elif metadata.content_type in [ContentType.IMAGE, ContentType.ARCHIVE]:
                    content = content_bytes
                else:
                    try:
                        content = content_bytes.decode('utf-8')
                    except:
                        content = content_bytes

                if update_access:
                    metadata.accessed_at = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
                    metadata.access_count += 1
                    self._save_registry()
                return metadata, content
            except Exception as e:
                logger.error(f"Failed to retrieve content {content_id}: {e}")
                return None

    def get_content_bytes(self, content_id: str) -> Optional[bytes]:
        """Get raw content bytes."""
        result = self.get_content(content_id)
        if not result:
            return None
        metadata, content = result
        if isinstance(content, bytes):
            return content
        if isinstance(content, str):
            return content.encode('utf-8')
        return json.dumps(content, default=str).encode('utf-8')

    def get_metadata(self, content_id: str) -> Optional[UnifiedContentMetadata]:
        """Get metadata without content."""
        return self._registry.get(content_id)

    def get_by_thread(self, thread_id: str) -> List[UnifiedContentMetadata]:
        """Get all content for a thread."""
        return [m for m in self._registry.values() if m.thread_id == thread_id]

    def get_by_user(self, user_id: str) -> List[UnifiedContentMetadata]:
        """Get all content for a user."""
        return [m for m in self._registry.values() if m.user_id == user_id]

    def delete_content(self, content_id: str) -> bool:
        """Delete content from disk and registry."""
        with self._lock:
            if content_id not in self._registry:
                return False
            metadata = self._registry[content_id]
            try:
                if os.path.exists(metadata.storage_path):
                    os.remove(metadata.storage_path)
            except Exception as e:
                logger.error(f"Failed to delete file {content_id}: {e}")
            del self._registry[content_id]
            self._save_registry()
            return True

    def cleanup_expired(self) -> int:
        """Remove expired content."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        expired_ids = [
            cid for cid, m in self._registry.items()
            if m.expires_at and now > datetime.fromisoformat(m.expires_at)
        ]
        for cid in expired_ids:
            self.delete_content(cid)
        return len(expired_ids)

    async def cleanup_expired_content(self) -> Dict[str, Any]:
        """Garbage collection with stats."""
        now = time.time()
        expired_ids = []
        bytes_freed = 0

        with self._lock:
            for cid, meta in self._registry.items():
                if meta.expires_at and meta.expires_at < now:
                    expired_ids.append(cid)

            for cid in expired_ids:
                meta = self._registry[cid]
                file_path = Path(meta.storage_path)
                if file_path.exists():
                    try:
                        bytes_freed += file_path.stat().st_size
                        os.remove(file_path)
                    except Exception as e:
                        logger.error(f"Failed to delete expired file {file_path}: {e}")
                del self._registry[cid]

            if expired_ids:
                self._save_registry()
                logger.info(f"🧹 GC Cleanup: Removed {len(expired_ids)} files, freed {bytes_freed/1024:.2f} KB")

        return {
            "deleted_count": len(expired_ids),
            "bytes_freed": bytes_freed,
            "expired_ids": expired_ids
        }

    def get_optimized_context(self, thread_id: str, max_tokens: int = 8000) -> Dict[str, Any]:
        """
        Build optimized context string for LLM by synthesizing active state and archived summaries.
        """
        all_metadata = self.get_by_thread(thread_id)
        if not all_metadata:
            return {
                "context_string": "No historical context available.",
                "references": [],
                "tokens_saved": 0
            }

        all_metadata.sort(key=lambda x: x.created_at)

        context_parts = []
        references = []
        tokens_saved = 0

        for meta in all_metadata:
            is_archive = "archive" in meta.tags
            ref = meta.to_reference()

            if is_archive:
                # Archives are just references
                context_parts.append(f"[ARCHIVE:{meta.id}] {meta.name}: {meta.summary}")
                references.append(ref.to_dict())
                tokens_saved += meta.original_size // 4 if meta.original_size else 0
            else:
                # Active artifacts/results
                context_parts.append(f"[ARTIFACT:{meta.id}] {meta.name}: {meta.summary}")
                references.append(ref.to_dict())

        context_string = "\n".join(context_parts)

        # Trim if too long
        max_chars = max_tokens * self.CHARS_PER_TOKEN_EST
        if len(context_string) > max_chars:
            context_string = context_string[:max_chars] + "... (truncated)"

        return {
            "context_string": context_string,
            "references": references,
            "tokens_saved": tokens_saved
        }

    async def register_user_upload(
        self,
        file_content: bytes,
        filename: str,
        user_id: str,
        thread_id: Optional[str] = None,
        mime_type: Optional[str] = None
    ) -> UnifiedContentMetadata:
        """Register a user-uploaded file."""
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
        content: Union[bytes, str],
        name: str,
        agent_id: str,
        user_id: str,
        thread_id: Optional[str] = None
    ) -> UnifiedContentMetadata:
        """Register agent output."""
        return await self.register_content(
            content=content,
            name=name,
            source=ContentSource.AGENT_OUTPUT,
            user_id=user_id,
            thread_id=thread_id,
            priority=ContentPriority.MEDIUM,
            tags=[f"agent:{agent_id}"]
        )

    async def register_artifact(
        self,
        content: Any,
        name: str,
        content_type: ContentType,
        thread_id: str,
        priority: ContentPriority = ContentPriority.MEDIUM,
        description: Optional[str] = None
    ) -> UnifiedContentMetadata:
        """Register a system artifact."""
        return await self.register_content(
            content=content,
            name=name,
            source=ContentSource.SYSTEM_GENERATED,
            content_type=content_type,
            thread_id=thread_id,
            priority=priority,
            tags=["artifact"],
            is_artifact=True,
            summary=description
        )

    async def upload_to_agent(
        self,
        content_id: str,
        agent_id: str,
        agent_base_url: str
    ) -> Optional[str]:
        """Upload content to an agent and cache the mapping."""
        if not agent_base_url:
            return None

        with self._lock:
            if content_id not in self._registry:
                return None
            meta = self._registry[content_id]

            # Check existing mapping
            if agent_id in meta.agent_mappings:
                mapping = meta.agent_mappings[agent_id]
                if mapping.is_valid:
                    return mapping.agent_content_id

        # Get content bytes
        content_bytes = self.get_content_bytes(content_id)
        if not content_bytes:
            return None

        # Upload using httpx
        import httpx
        upload_url = f"{agent_base_url.rstrip('/')}/upload"

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                files = {"file": (meta.name, content_bytes, meta.mime_type)}
                response = await client.post(upload_url, files=files)

                if response.status_code == 200:
                    result = response.json()
                    agent_content_id = None
                    if isinstance(result, dict):
                        if 'result' in result and isinstance(result['result'], dict):
                            agent_content_id = result['result'].get('file_id')
                        elif 'file_id' in result:
                            agent_content_id = result['file_id']
                        elif 'id' in result:
                            agent_content_id = result['id']

                    if agent_content_id:
                        with self._lock:
                            meta = self._registry[content_id]
                            meta.agent_mappings[agent_id] = AgentContentMapping(
                                content_id=content_id,
                                agent_id=agent_id,
                                agent_content_id=agent_content_id,
                                agent_endpoint=upload_url
                            )
                            self._save_registry()
                        return agent_content_id
                    else:
                        logger.warning(f"Could not extract ID from upload response: {result}")
                else:
                    logger.error(f"Agent upload failed: {response.status_code} {response.text}")

        except Exception as e:
            logger.error(f"Exception uploading to agent: {e}")

        return None


# Singleton instance
_cms_instance: Optional[ContentManagementService] = None


def get_cms() -> ContentManagementService:
    """Get or create CMS singleton."""
    global _cms_instance
    if _cms_instance is None:
        _cms_instance = ContentManagementService()
    return _cms_instance
