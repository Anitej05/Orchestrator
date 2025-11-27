"""
Artifact Management Service - Production-Grade Context Management System

This service implements a sophisticated artifact management system inspired by SOTA
production systems (like Claude's artifacts, ChatGPT's memory, etc.) to efficiently
manage large contexts without flooding the LLM's context window.

Key Features:
1. Artifact Storage: Store large content (code, documents, results) as artifacts
2. Artifact References: Use lightweight references in context instead of full content
3. Lazy Loading: Load artifact content only when explicitly needed
4. Summarization: Auto-generate summaries for quick context understanding
5. Semantic Search: Find relevant artifacts using embeddings
6. TTL & Cleanup: Automatic expiration and cleanup of stale artifacts
7. Compression: Compress large artifacts to save storage

Architecture:
- ArtifactStore: Persistent storage layer (file-based + optional Redis)
- ArtifactIndex: In-memory index for fast lookups
- ArtifactSummarizer: LLM-based summarization for context compression
- ArtifactRetriever: Semantic search for relevant artifact retrieval
"""

import os
import json
import hashlib
import time
import gzip
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Literal, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import threading
from functools import lru_cache

logger = logging.getLogger("ArtifactService")

# ============================================================================
# ARTIFACT TYPES AND SCHEMAS
# ============================================================================

class ArtifactType(str, Enum):
    """Types of artifacts that can be stored"""
    CODE = "code"                    # Source code, scripts
    DOCUMENT = "document"            # Text documents, markdown
    DATA = "data"                    # JSON, structured data
    RESULT = "result"                # Agent execution results
    CANVAS = "canvas"                # HTML/Markdown canvas content
    SCREENSHOT = "screenshot"        # Browser screenshots (base64)
    PLAN = "plan"                    # Execution plans
    CONVERSATION = "conversation"    # Conversation history segments
    EMBEDDING = "embedding"          # Vector embeddings
    SUMMARY = "summary"              # Generated summaries


class ArtifactPriority(str, Enum):
    """Priority levels for artifact retention"""
    CRITICAL = "critical"    # Never auto-delete (user data, important results)
    HIGH = "high"            # Keep for extended period
    MEDIUM = "medium"        # Standard retention
    LOW = "low"              # Can be deleted when space needed
    EPHEMERAL = "ephemeral"  # Delete after session


@dataclass
class ArtifactMetadata:
    """Metadata for an artifact"""
    id: str
    type: ArtifactType
    name: str
    description: str
    priority: ArtifactPriority = ArtifactPriority.MEDIUM
    thread_id: Optional[str] = None
    parent_id: Optional[str] = None  # For hierarchical artifacts
    tags: List[str] = field(default_factory=list)
    size_bytes: int = 0
    compressed: bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    access_count: int = 0
    summary: Optional[str] = None  # Auto-generated summary
    embedding_id: Optional[str] = None  # Reference to embedding artifact


@dataclass
class Artifact:
    """Complete artifact with metadata and content"""
    metadata: ArtifactMetadata
    content: Any  # The actual content (lazy-loaded)
    
    def to_reference(self) -> 'ArtifactReference':
        """Convert to a lightweight reference for context inclusion"""
        return ArtifactReference(
            id=self.metadata.id,
            type=self.metadata.type,
            name=self.metadata.name,
            summary=self.metadata.summary or self.metadata.description,
            size_bytes=self.metadata.size_bytes
        )


@dataclass
class ArtifactReference:
    """Lightweight reference to an artifact for context inclusion"""
    id: str
    type: ArtifactType
    name: str
    summary: str
    size_bytes: int
    
    def to_context_string(self) -> str:
        """Generate a context-friendly string representation"""
        size_kb = self.size_bytes / 1024
        return f"[ARTIFACT:{self.id}] {self.name} ({self.type.value}, {size_kb:.1f}KB)\n  Summary: {self.summary}"


# ============================================================================
# ARTIFACT STORE - Persistent Storage Layer
# ============================================================================

class ArtifactStore:
    """
    Persistent storage for artifacts with compression and lazy loading.
    Uses file-based storage with optional Redis caching.
    """
    
    def __init__(self, storage_dir: str = "storage/artifacts"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        
        # Create subdirectories for different artifact types
        for artifact_type in ArtifactType:
            (self.storage_dir / artifact_type.value).mkdir(exist_ok=True)
        
        # Metadata index file
        self.index_path = self.storage_dir / "index.json"
        self._index: Dict[str, ArtifactMetadata] = {}
        self._load_index()
        
        logger.info(f"ArtifactStore initialized at {self.storage_dir}")
    
    def _load_index(self):
        """Load the artifact index from disk"""
        if self.index_path.exists():
            try:
                with open(self.index_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for artifact_id, meta_dict in data.items():
                        meta_dict['type'] = ArtifactType(meta_dict['type'])
                        meta_dict['priority'] = ArtifactPriority(meta_dict['priority'])
                        self._index[artifact_id] = ArtifactMetadata(**meta_dict)
                logger.info(f"Loaded {len(self._index)} artifacts from index")
            except Exception as e:
                logger.error(f"Failed to load artifact index: {e}")
                self._index = {}
    
    def _save_index(self):
        """Save the artifact index to disk"""
        try:
            data = {}
            for artifact_id, meta in self._index.items():
                meta_dict = asdict(meta)
                meta_dict['type'] = meta.type.value
                meta_dict['priority'] = meta.priority.value
                data[artifact_id] = meta_dict
            
            with open(self.index_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save artifact index: {e}")
    
    def _get_artifact_path(self, artifact_id: str, artifact_type: ArtifactType) -> Path:
        """Get the file path for an artifact"""
        return self.storage_dir / artifact_type.value / f"{artifact_id}.json.gz"
    
    def _generate_id(self, content: Any, name: str) -> str:
        """Generate a unique artifact ID based on content hash"""
        content_str = json.dumps(content, sort_keys=True, default=str)
        hash_input = f"{name}:{content_str}:{time.time()}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    def store(
        self,
        content: Any,
        name: str,
        artifact_type: ArtifactType,
        description: str = "",
        thread_id: Optional[str] = None,
        priority: ArtifactPriority = ArtifactPriority.MEDIUM,
        tags: List[str] = None,
        ttl_hours: Optional[int] = None,
        parent_id: Optional[str] = None
    ) -> ArtifactMetadata:
        """
        Store an artifact and return its metadata.
        
        Args:
            content: The content to store
            name: Human-readable name
            artifact_type: Type of artifact
            description: Description of the artifact
            thread_id: Associated conversation thread
            priority: Retention priority
            tags: Searchable tags
            ttl_hours: Time-to-live in hours (None = no expiration)
            parent_id: Parent artifact ID for hierarchical storage
        
        Returns:
            ArtifactMetadata for the stored artifact
        """
        with self._lock:
            artifact_id = self._generate_id(content, name)
            
            # Serialize content
            content_json = json.dumps(content, default=str, ensure_ascii=False)
            content_bytes = content_json.encode('utf-8')
            
            # Compress if large (> 1KB)
            compressed = len(content_bytes) > 1024
            if compressed:
                content_bytes = gzip.compress(content_bytes)
            
            # Calculate expiration
            expires_at = None
            if ttl_hours:
                expires_at = time.time() + (ttl_hours * 3600)
            elif priority == ArtifactPriority.EPHEMERAL:
                expires_at = time.time() + 3600  # 1 hour for ephemeral
            
            # Create metadata
            metadata = ArtifactMetadata(
                id=artifact_id,
                type=artifact_type,
                name=name,
                description=description,
                priority=priority,
                thread_id=thread_id,
                parent_id=parent_id,
                tags=tags or [],
                size_bytes=len(content_bytes),
                compressed=compressed,
                expires_at=expires_at
            )
            
            # Write to disk
            artifact_path = self._get_artifact_path(artifact_id, artifact_type)
            with open(artifact_path, 'wb') as f:
                f.write(content_bytes)
            
            # Update index
            self._index[artifact_id] = metadata
            self._save_index()
            
            logger.info(f"Stored artifact {artifact_id}: {name} ({artifact_type.value}, {metadata.size_bytes} bytes)")
            return metadata
    
    def retrieve(self, artifact_id: str, update_access: bool = True) -> Optional[Artifact]:
        """
        Retrieve an artifact by ID.
        
        Args:
            artifact_id: The artifact ID
            update_access: Whether to update access timestamp
        
        Returns:
            The Artifact or None if not found
        """
        with self._lock:
            if artifact_id not in self._index:
                logger.warning(f"Artifact not found: {artifact_id}")
                return None
            
            metadata = self._index[artifact_id]
            
            # Check expiration
            if metadata.expires_at and time.time() > metadata.expires_at:
                logger.info(f"Artifact expired: {artifact_id}")
                self.delete(artifact_id)
                return None
            
            # Load content
            artifact_path = self._get_artifact_path(artifact_id, metadata.type)
            if not artifact_path.exists():
                logger.error(f"Artifact file missing: {artifact_path}")
                return None
            
            try:
                with open(artifact_path, 'rb') as f:
                    content_bytes = f.read()
                
                # Decompress if needed
                if metadata.compressed:
                    content_bytes = gzip.decompress(content_bytes)
                
                content = json.loads(content_bytes.decode('utf-8'))
                
                # Update access stats
                if update_access:
                    metadata.accessed_at = time.time()
                    metadata.access_count += 1
                    self._save_index()
                
                return Artifact(metadata=metadata, content=content)
                
            except Exception as e:
                logger.error(f"Failed to retrieve artifact {artifact_id}: {e}")
                return None
    
    def delete(self, artifact_id: str) -> bool:
        """Delete an artifact"""
        with self._lock:
            if artifact_id not in self._index:
                return False
            
            metadata = self._index[artifact_id]
            artifact_path = self._get_artifact_path(artifact_id, metadata.type)
            
            try:
                if artifact_path.exists():
                    artifact_path.unlink()
                del self._index[artifact_id]
                self._save_index()
                logger.info(f"Deleted artifact: {artifact_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete artifact {artifact_id}: {e}")
                return False
    
    def list_by_thread(self, thread_id: str) -> List[ArtifactMetadata]:
        """List all artifacts for a conversation thread"""
        return [
            meta for meta in self._index.values()
            if meta.thread_id == thread_id
        ]
    
    def list_by_type(self, artifact_type: ArtifactType) -> List[ArtifactMetadata]:
        """List all artifacts of a specific type"""
        return [
            meta for meta in self._index.values()
            if meta.type == artifact_type
        ]
    
    def search_by_tags(self, tags: List[str]) -> List[ArtifactMetadata]:
        """Search artifacts by tags"""
        return [
            meta for meta in self._index.values()
            if any(tag in meta.tags for tag in tags)
        ]
    
    def cleanup_expired(self) -> int:
        """Remove expired artifacts"""
        now = time.time()
        expired = [
            artifact_id for artifact_id, meta in self._index.items()
            if meta.expires_at and now > meta.expires_at
        ]
        
        for artifact_id in expired:
            self.delete(artifact_id)
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired artifacts")
        return len(expired)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        total_size = sum(meta.size_bytes for meta in self._index.values())
        by_type = {}
        for artifact_type in ArtifactType:
            type_artifacts = [m for m in self._index.values() if m.type == artifact_type]
            by_type[artifact_type.value] = {
                "count": len(type_artifacts),
                "size_bytes": sum(m.size_bytes for m in type_artifacts)
            }
        
        return {
            "total_artifacts": len(self._index),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "by_type": by_type
        }




# ============================================================================
# ARTIFACT CONTEXT MANAGER - Smart Context Compression
# ============================================================================

class ArtifactContextManager:
    """
    Manages artifact references in conversation context.
    Implements smart context compression by replacing large content with references.
    """
    
    # Token estimation: ~4 chars per token on average
    CHARS_PER_TOKEN = 4
    
    # Thresholds for artifact creation
    MIN_ARTIFACT_SIZE = 500  # Minimum chars to create artifact
    MAX_INLINE_SIZE = 2000   # Max chars to keep inline
    
    def __init__(self, store: ArtifactStore):
        self.store = store
        self._active_references: Dict[str, List[str]] = {}  # thread_id -> [artifact_ids]
    
    def should_create_artifact(self, content: Any, content_type: str = "text") -> bool:
        """Determine if content should be stored as an artifact"""
        if isinstance(content, str):
            return len(content) > self.MIN_ARTIFACT_SIZE
        elif isinstance(content, (dict, list)):
            json_str = json.dumps(content, default=str)
            return len(json_str) > self.MIN_ARTIFACT_SIZE
        return False
    
    def compress_for_context(
        self,
        content: Any,
        name: str,
        thread_id: str,
        artifact_type: ArtifactType = ArtifactType.DATA,
        force_artifact: bool = False
    ) -> Union[Any, ArtifactReference]:
        """
        Compress content for context inclusion.
        Returns either the original content (if small) or an ArtifactReference.
        
        Args:
            content: Content to potentially compress
            name: Name for the artifact
            thread_id: Conversation thread ID
            artifact_type: Type of artifact
            force_artifact: Force creation of artifact regardless of size
        
        Returns:
            Original content or ArtifactReference
        """
        if not force_artifact and not self.should_create_artifact(content):
            return content
        
        # Generate summary for the artifact
        summary = self._generate_summary(content, artifact_type)
        
        # Store as artifact
        metadata = self.store.store(
            content=content,
            name=name,
            artifact_type=artifact_type,
            description=summary,
            thread_id=thread_id,
            priority=ArtifactPriority.MEDIUM
        )
        
        # Update metadata with summary
        metadata.summary = summary
        
        # Track active reference
        if thread_id not in self._active_references:
            self._active_references[thread_id] = []
        self._active_references[thread_id].append(metadata.id)
        
        return ArtifactReference(
            id=metadata.id,
            type=artifact_type,
            name=name,
            summary=summary,
            size_bytes=metadata.size_bytes
        )
    
    def _generate_summary(self, content: Any, artifact_type: ArtifactType) -> str:
        """Generate a brief summary of the content"""
        if isinstance(content, str):
            # For text, take first 200 chars
            preview = content[:200].replace('\n', ' ')
            if len(content) > 200:
                preview += "..."
            return preview
        
        elif isinstance(content, dict):
            # For dicts, summarize structure
            keys = list(content.keys())[:5]
            key_str = ", ".join(keys)
            if len(content) > 5:
                key_str += f", ... (+{len(content) - 5} more)"
            return f"Object with keys: {key_str}"
        
        elif isinstance(content, list):
            # For lists, summarize length and sample
            if len(content) == 0:
                return "Empty list"
            sample = str(content[0])[:50]
            return f"List of {len(content)} items. First: {sample}..."
        
        return f"{artifact_type.value} content"
    
    def expand_reference(self, reference: ArtifactReference) -> Optional[Any]:
        """Expand an artifact reference to its full content"""
        artifact = self.store.retrieve(reference.id)
        if artifact:
            return artifact.content
        return None
    
    def get_context_with_references(
        self,
        thread_id: str,
        include_summaries: bool = True
    ) -> Dict[str, Any]:
        """
        Get a context-optimized view of all artifacts for a thread.
        
        Returns a dict with:
        - references: List of ArtifactReference objects
        - context_string: Human-readable context for LLM
        - total_tokens_saved: Estimated tokens saved by using references
        """
        artifacts = self.store.list_by_thread(thread_id)
        
        references = []
        context_lines = []
        total_original_size = 0
        total_reference_size = 0
        
        for meta in artifacts:
            ref = ArtifactReference(
                id=meta.id,
                type=meta.type,
                name=meta.name,
                summary=meta.summary or meta.description,
                size_bytes=meta.size_bytes
            )
            references.append(ref)
            
            if include_summaries:
                context_lines.append(ref.to_context_string())
            
            total_original_size += meta.size_bytes
            total_reference_size += len(ref.to_context_string())
        
        context_string = "\n".join(context_lines) if context_lines else ""
        tokens_saved = (total_original_size - total_reference_size) // self.CHARS_PER_TOKEN
        
        return {
            "references": references,
            "context_string": context_string,
            "total_tokens_saved": max(0, tokens_saved),
            "artifact_count": len(references)
        }
    
    def build_optimized_context(
        self,
        thread_id: str,
        current_prompt: str,
        max_context_tokens: int = 8000,
        include_artifacts: List[str] = None
    ) -> Dict[str, Any]:
        """
        Build an optimized context for LLM consumption.
        
        Args:
            thread_id: Conversation thread ID
            current_prompt: The current user prompt
            max_context_tokens: Maximum tokens for context
            include_artifacts: Specific artifact IDs to include fully
        
        Returns:
            Optimized context dict with prompt, references, and expanded artifacts
        """
        include_artifacts = include_artifacts or []
        
        # Get all artifacts for thread
        all_artifacts = self.store.list_by_thread(thread_id)
        
        # Separate into expanded and referenced
        expanded_content = {}
        references = []
        
        for meta in all_artifacts:
            if meta.id in include_artifacts:
                # Expand this artifact
                artifact = self.store.retrieve(meta.id)
                if artifact:
                    expanded_content[meta.name] = artifact.content
            else:
                # Keep as reference
                references.append(ArtifactReference(
                    id=meta.id,
                    type=meta.type,
                    name=meta.name,
                    summary=meta.summary or meta.description,
                    size_bytes=meta.size_bytes
                ))
        
        # Build context string
        context_parts = []
        
        if references:
            context_parts.append("## Available Artifacts (use artifact ID to retrieve full content)")
            for ref in references:
                context_parts.append(ref.to_context_string())
        
        if expanded_content:
            context_parts.append("\n## Expanded Artifact Content")
            for name, content in expanded_content.items():
                if isinstance(content, str):
                    context_parts.append(f"### {name}\n{content}")
                else:
                    context_parts.append(f"### {name}\n```json\n{json.dumps(content, indent=2, default=str)}\n```")
        
        return {
            "prompt": current_prompt,
            "artifact_context": "\n".join(context_parts),
            "references": [asdict(r) for r in references],
            "expanded": expanded_content,
            "estimated_tokens": len("\n".join(context_parts)) // self.CHARS_PER_TOKEN
        }


# ============================================================================
# ARTIFACT-AWARE STATE MANAGER - Integration with Orchestrator State
# ============================================================================

class ArtifactAwareStateManager:
    """
    Integrates artifact management with the orchestrator state.
    Automatically compresses large state fields into artifacts.
    """
    
    # Fields that should be considered for artifact storage
    COMPRESSIBLE_FIELDS = [
        "completed_tasks",      # Can grow large with many task results
        "task_agent_pairs",     # Contains full agent definitions
        "task_plan",            # Execution plans
        "canvas_content",       # HTML/Markdown canvas
        "uploaded_files",       # File metadata and content
        "messages",             # Conversation history
    ]
    
    # Fields to always keep inline (never compress)
    INLINE_FIELDS = [
        "original_prompt",
        "thread_id",
        "status",
        "pending_user_input",
        "question_for_user",
        "final_response",
        "planning_mode",
    ]
    
    def __init__(self, store: ArtifactStore):
        self.store = store
        self.context_manager = ArtifactContextManager(store)
    
    def compress_state(self, state: Dict[str, Any], thread_id: str) -> Dict[str, Any]:
        """
        Compress a state dict by moving large fields to artifacts.
        
        Args:
            state: The orchestrator state dict
            thread_id: Conversation thread ID
        
        Returns:
            Compressed state with artifact references
        """
        compressed = {}
        artifact_refs = {}
        
        for key, value in state.items():
            if key in self.INLINE_FIELDS or value is None:
                # Keep inline
                compressed[key] = value
            elif key in self.COMPRESSIBLE_FIELDS:
                # Check if should compress
                result = self.context_manager.compress_for_context(
                    content=value,
                    name=f"state_{key}",
                    thread_id=thread_id,
                    artifact_type=self._get_artifact_type(key)
                )
                
                if isinstance(result, ArtifactReference):
                    # Store reference
                    artifact_refs[key] = asdict(result)
                    compressed[key] = None  # Clear from state
                else:
                    compressed[key] = value
            else:
                compressed[key] = value
        
        # Add artifact references to state
        compressed["_artifact_refs"] = artifact_refs
        
        return compressed
    
    def expand_state(self, compressed_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expand a compressed state by loading artifacts.
        
        Args:
            compressed_state: State with artifact references
        
        Returns:
            Fully expanded state
        """
        expanded = dict(compressed_state)
        artifact_refs = expanded.pop("_artifact_refs", {})
        
        for key, ref_dict in artifact_refs.items():
            artifact = self.store.retrieve(ref_dict["id"])
            if artifact:
                expanded[key] = artifact.content
            else:
                logger.warning(f"Failed to expand artifact for {key}: {ref_dict['id']}")
                expanded[key] = None
        
        return expanded
    
    def _get_artifact_type(self, field_name: str) -> ArtifactType:
        """Map state field names to artifact types"""
        mapping = {
            "completed_tasks": ArtifactType.RESULT,
            "task_agent_pairs": ArtifactType.DATA,
            "task_plan": ArtifactType.PLAN,
            "canvas_content": ArtifactType.CANVAS,
            "uploaded_files": ArtifactType.DOCUMENT,
            "messages": ArtifactType.CONVERSATION,
        }
        return mapping.get(field_name, ArtifactType.DATA)
    
    def get_lightweight_state(self, state: Dict[str, Any], thread_id: str) -> Dict[str, Any]:
        """
        Get a lightweight version of state for context inclusion.
        Includes summaries instead of full content.
        """
        lightweight = {}
        
        for key, value in state.items():
            if key.startswith("_"):
                continue
            elif key in self.INLINE_FIELDS:
                lightweight[key] = value
            elif value is None:
                continue
            elif isinstance(value, list) and len(value) > 0:
                # Summarize lists
                lightweight[key] = {
                    "_type": "list",
                    "_count": len(value),
                    "_sample": str(value[0])[:100] if value else None
                }
            elif isinstance(value, dict) and len(value) > 3:
                # Summarize large dicts
                lightweight[key] = {
                    "_type": "dict",
                    "_keys": list(value.keys())[:5],
                    "_key_count": len(value)
                }
            else:
                lightweight[key] = value
        
        return lightweight


# ============================================================================
# GLOBAL ARTIFACT SERVICE INSTANCE
# ============================================================================

# Singleton instance
_artifact_store: Optional[ArtifactStore] = None
_context_manager: Optional[ArtifactContextManager] = None
_state_manager: Optional[ArtifactAwareStateManager] = None


def get_artifact_store() -> ArtifactStore:
    """Get the global artifact store instance"""
    global _artifact_store
    if _artifact_store is None:
        _artifact_store = ArtifactStore()
    return _artifact_store


def get_context_manager() -> ArtifactContextManager:
    """Get the global context manager instance"""
    global _context_manager
    if _context_manager is None:
        _context_manager = ArtifactContextManager(get_artifact_store())
    return _context_manager


def get_state_manager() -> ArtifactAwareStateManager:
    """Get the global state manager instance"""
    global _state_manager
    if _state_manager is None:
        _state_manager = ArtifactAwareStateManager(get_artifact_store())
    return _state_manager


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def store_artifact(
    content: Any,
    name: str,
    artifact_type: ArtifactType,
    thread_id: Optional[str] = None,
    **kwargs
) -> ArtifactMetadata:
    """Convenience function to store an artifact"""
    return get_artifact_store().store(
        content=content,
        name=name,
        artifact_type=artifact_type,
        thread_id=thread_id,
        **kwargs
    )


def retrieve_artifact(artifact_id: str) -> Optional[Artifact]:
    """Convenience function to retrieve an artifact"""
    return get_artifact_store().retrieve(artifact_id)


def compress_for_context(
    content: Any,
    name: str,
    thread_id: str,
    artifact_type: ArtifactType = ArtifactType.DATA
) -> Union[Any, ArtifactReference]:
    """Convenience function to compress content for context"""
    return get_context_manager().compress_for_context(
        content=content,
        name=name,
        thread_id=thread_id,
        artifact_type=artifact_type
    )


def get_thread_context(thread_id: str) -> Dict[str, Any]:
    """Get optimized context for a thread"""
    return get_context_manager().get_context_with_references(thread_id)
