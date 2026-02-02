"""
Content Management Service

This service consolidates file management, artifact management, and intelligent content processing
into a single, standardized system. It features a Map-Reduce engine for handling large context
tasks (summarization, search, analysis) using specific LLMs.
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
import math
import time
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import io

try:
    import pandas as pd
except ImportError:
    pd = None

# Local imports
# Adjust as needed if these are used elsewhere
from agents.utils.standard_file_interface import AgentFileMetadata

# Robust Import for KeyManager (Handles potential shadowing)
import sys
orbimesh_root = Path(__file__).parent.parent.parent.resolve() # services -> backend -> Orbimesh
if str(orbimesh_root) not in sys.path:
    sys.path.insert(0, str(orbimesh_root))

from backend.utils.key_manager import get_cerebras_key, report_rate_limit

logger = logging.getLogger("ContentManagementService")

# Storage directories
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
STORAGE_BASE = PROJECT_ROOT / "storage"
SYSTEM_DIR = STORAGE_BASE / "system"
CONTENT_DIR = SYSTEM_DIR / "content"

USER_UPLOADS_DIR = CONTENT_DIR / "uploads"
AGENT_FILES_DIR = CONTENT_DIR / "agent_files"
ARTIFACTS_DIR = CONTENT_DIR / "artifacts"
TEMP_DIR = CONTENT_DIR / "temp"

# Ensure directories exist
SYSTEM_DIR.mkdir(parents=True, exist_ok=True)
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

class ProcessingTaskType(str, Enum):
    SUMMARIZE = "summarize"
    SEARCH = "search"
    EXTRACT = "extract"
    ANALYZE = "analyze"


class ProcessingStrategy(str, Enum):
    STANDARD = "standard"  # Just process and return result
    CONTEXT_OPTIMIZATION = "context_optimization"  # Chunk, Summarize, Archive Chunks, Return Summary + Refs
    COMPREHENSIVE_AUDIT = "comprehensive_audit"  # Deep analysis, save intermediate steps
    DATA_EXTRACTION = "data_extraction"  # Extract structured data
    ARCHIVAL_MEMORY = "archival_memory"  # Archive older conversation turns + Summarize


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


@dataclass
class ProcessingResult:
    """Result of a map-reduce processing operation"""
    task_type: ProcessingTaskType
    final_output: str
    chunk_count: int
    processing_time_ms: float
    model_map: str
    model_reduce: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContentManagementService:
    """
    Central service for managing all content across the orchestrator and agents.
    
    Features:
    - Unified file/artifact registry
    - Intelligent Map-Reduce for large content processing
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

    # Map-Reduce Configuration
    CHUNK_SIZE_TOKENS = 8000  # Conservative estimate for helper models
    CHARS_PER_TOKEN_EST = 4   # Rough approximation
    CHUNK_SIZE_CHARS = CHUNK_SIZE_TOKENS * CHARS_PER_TOKEN_EST
    
    # Models
    # Models
    MODEL_HELPER = "llama-3.3-70b"  # For Map phase
    MODEL_GENERATOR = "gpt-oss-120b" # For Reduce phase
    
    def __init__(self, storage_dir: str = "storage/content"):
        self._registry: Dict[str, UnifiedContentMetadata] = {}
        self._registry_path = Path(storage_dir) / "content_registry.json"
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._load_registry()
        
        # Initialize LLM client lazily
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
        if ext in ['.csv', '.xlsx', '.xls', '.ods']: return ContentType.SPREADSHEET
        if mime_type and mime_type.startswith('image/'): return ContentType.IMAGE
        if ext in ['.pdf', '.doc', '.docx', '.txt', '.md', '.rtf']: return ContentType.DOCUMENT
        if ext in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.go', '.rs']: return ContentType.CODE
        if ext in ['.json', '.xml', '.yaml', '.yml']: return ContentType.DATA
        if ext in ['.zip', '.tar', '.gz', '.rar', '.7z']: return ContentType.ARCHIVE
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
            ContentPriority.CRITICAL: None,
            ContentPriority.HIGH: 30 * 24,
            ContentPriority.MEDIUM: 7 * 24,
            ContentPriority.LOW: 24,
            ContentPriority.EPHEMERAL: 1,
        }
        hours = ttl_map.get(priority)
        if hours:
            return (datetime.utcnow() + timedelta(hours=hours)).isoformat()
        return None
    
    def _generate_summary(self, content: Any, content_type: ContentType) -> str:
        """Generate a brief summary of the content (simple preview)"""
        if isinstance(content, bytes):
            return f"Binary content ({len(content)} bytes)"
        if isinstance(content, str):
            preview = content[:200].replace('\n', ' ')
            if len(content) > 200: preview += "..."
            return preview
        if isinstance(content, dict):
            return f"Object with keys: {list(content.keys())[:5]}..."
        if isinstance(content, list):
            return f"List of {len(content)} items..."
        return f"{content_type.value} content"

    # =========================================================================
    # MAP-REDUCE PROCESSING ENGINE
    # =========================================================================

    def _get_llm_client(self, model_name: str):
        """Get or initialize LLM client for specific model"""
        try:
            from langchain_cerebras import ChatCerebras
            # Use KeyManager (blocking wait if needed)
            api_key = get_cerebras_key()
            if not api_key:
                logger.warning("No Cerebras API key available from manager")
                return None
            return ChatCerebras(model=model_name, api_key=api_key)
        except ImportError:
            logger.error("langchain_cerebras not installed")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize LLM {model_name}: {e}")
            return None

    async def _invoke_with_retry(self, llm_client, prompt: str, max_retries: int = 3) -> Any:
        """
        Robust invocation wrapper with Key Rotation support.
        Only rotates if the client is ChatCerebras and hits a rate limit.
        """
        current_llm = llm_client
        for attempt in range(max_retries):
            try:
                # Direct ainvoke
                return await current_llm.ainvoke(prompt)
            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = any(k in error_str for k in ['429', '413', 'rate_limit', 'too many requests'])
                if hasattr(e, 'response') and hasattr(e.response, 'status_code') and e.response.status_code == 429:
                    is_rate_limit = True
                
                # Check if it's Cerebras
                is_cerebras = "cerebras" in str(type(current_llm)).lower()
                
                if is_cerebras and is_rate_limit:
                    logger.warning(f"⚡ MapReduce Rate Limit (Attempt {attempt+1}). Rotating...")
                    
                    # Report limit
                    try:
                        if hasattr(current_llm, 'api_key'):
                            report_rate_limit(current_llm.api_key)
                    except:
                        pass
                        
                    # Get new key & Re-init
                    new_key = get_cerebras_key()
                    from langchain_cerebras import ChatCerebras
                    # Assuming we know the model... tricky. 
                    # We can try to extract it from the old client or default to helper/generator depending on context?
                    # Or simpler: just re-ask _get_llm_client for the *same model*?
                    # But we don't know the model name here easily.
                    # We'll just instantiate a new client with the same config if possible.
                    # Hack: access .model_name or .model?
                    model = getattr(current_llm, 'model_name', getattr(current_llm, 'model', self.MODEL_HELPER))
                    
                    current_llm = ChatCerebras(model=model, api_key=new_key)
                    logger.info(f"🔄 Retrying with new key...")
                    continue
                
                if attempt == max_retries - 1:
                    raise
                # Exponential backoff for non-rate-limit temporary errors?
                await asyncio.sleep(1 * (attempt + 1))
        raise ValueError("Max retries exceeded")

    def _chunk_content(self, content: Any, content_type: ContentType, original_name: str = "content") -> List[str]:
        """
        Split content into manageable chunks based on type.
        Returns a list of text representations of chunks.
        """
        chunks = []
        
        # --- SPREADSHEET CHUNKING ---
        if content_type == ContentType.SPREADSHEET:
            if not pd:
                logger.warning("Pandas not available for spreadsheet chunking, falling back to text")
                return self._chunk_text(str(content))
            
            try:
                df = None
                if isinstance(content, bytes):
                    # Try Excel then CSV
                    try:
                        df = pd.read_excel(io.BytesIO(content))
                    except:
                        df = pd.read_csv(io.BytesIO(content))
                elif isinstance(content, str):
                    df = pd.read_csv(io.StringIO(content))
                
                if df is not None:
                    # Split by rows (e.g., 50 rows per chunk to keep context manageable)
                    ROWS_PER_CHUNK = 50
                    for i in range(0, len(df), ROWS_PER_CHUNK):
                        chunk_df = df.iloc[i:i+ROWS_PER_CHUNK]
                        # Convert to markdown table for LLM
                        chunks.append(chunk_df.to_markdown(index=False))
                    logger.info(f"Chunked spreadsheet {original_name} into {len(chunks)} chunks")
                    return chunks
            except Exception as e:
                logger.error(f"Failed to chunk spreadsheet: {e}")
                # Fallback to text chunking
        
        # --- CODE CHUNKING ---
        elif content_type == ContentType.CODE:
            # Simple line-based chunking preserving boundaries best we can
            # Future: AST-based splitting
            text = str(content)
            lines = text.split('\n')
            LINES_PER_CHUNK = 200
            
            current_chunk = []
            current_len = 0
            
            for line in lines:
                current_chunk.append(line)
                current_len += len(line)
                
                # Check soft limit, but try to break on empty lines or dedents (heuristic)
                if len(current_chunk) >= LINES_PER_CHUNK:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_len = 0
            
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
            return chunks

        # --- DEFAULT TEXT CHUNKING ---
        text_content = str(content)
        if isinstance(content, bytes):
            try:
                text_content = content.decode('utf-8')
            except:
                text_content = str(content) # Fallback repr
                
        return self._chunk_text(text_content)

    def _chunk_text(self, content: str) -> List[str]:
        """Token/Char optimized text splitter"""
        chunks = []
        current_pos = 0
        content_len = len(content)
        
        while current_pos < content_len:
            end_pos = min(current_pos + self.CHUNK_SIZE_CHARS, content_len)
            
            if end_pos < content_len:
                last_newline = content.rfind('\n', current_pos, end_pos)
                if last_newline != -1 and last_newline > current_pos + (self.CHUNK_SIZE_CHARS // 2):
                    end_pos = last_newline + 1
                else:
                    last_space = content.rfind(' ', current_pos, end_pos)
                    if last_space != -1 and last_space > current_pos + (self.CHUNK_SIZE_CHARS // 2):
                        end_pos = last_space + 1
            
            chunks.append(content[current_pos:end_pos])
            current_pos = end_pos
            
        return chunks

    async def _process_chunk_map(self, chunk: str, task_type: ProcessingTaskType, query: Optional[str] = None) -> str:
        """Helper Phase (Map): Process a single chunk"""
        llm = self._get_llm_client(self.MODEL_HELPER)
        if not llm:
            return "Error: LLM unavailable"

        # Define prompts based on task type
        if task_type == ProcessingTaskType.SUMMARIZE:
            prompt = f"""Summarize the following text chunk concisely. Capture key points/events.
Chunk:
{chunk[:20000]}... (truncated)

Summary:"""
        elif task_type == ProcessingTaskType.SEARCH:
            prompt = f"""Search this text chunk for information related to: "{query}".
If found, extract relevant details. If not found, reply "None".
Chunk:
{chunk[:20000]}... (truncated)

Result:"""
        elif task_type == ProcessingTaskType.EXTRACT:
             prompt = f"""Extract information related to "{query}" from this text.
Chunk:
{chunk[:20000]}... (truncated)

Extraction:"""
        else: # ANALYZE or default
            prompt = f"""Analyze this text chunk regarding: "{query}".
Chunk:
{chunk[:20000]}... (truncated)

Analysis:"""

        try:
            # Use retry wrapper
            response = await self._invoke_with_retry(llm, prompt)
            # Strip <think> tags if model is reasoning
            content = response.content
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            return content
        except Exception as e:
            logger.error(f"Map phase failed for chunk: {e}")
            return f"[Error processing chunk: {e}]"

    async def _process_reduce(self, mapped_results: List[str], task_type: ProcessingTaskType, query: Optional[str] = None) -> str:
        """Generator Phase (Reduce): Synthesize final answer from mapped results"""
        llm = self._get_llm_client(self.MODEL_GENERATOR)
        if not llm:
            return "Error: LLM unavailable"

        # Filter out empty/irrelevant results
        valid_results = [r for r in mapped_results if r and "None" not in r and "Error" not in r]
        
        if not valid_results:
            return "No relevant information found in the content."

        combined_context = "\n---\n".join(valid_results)
        
        # Define prompts
        if task_type == ProcessingTaskType.SUMMARIZE:
            prompt = f"""Synthesize these chunk summaries into a coherent, comprehensive final summary.
Context:
{combined_context}

Final Summary:"""
        elif task_type == ProcessingTaskType.SEARCH:
            prompt = f"""Synthesize these search findings for the query: "{query}".
Context:
{combined_context}

Final Answer:"""
        else:
            prompt = f"""Synthesize these analysis points into a final report regarding: "{query}".
Context:
{combined_context}

Final Report:"""

        try:
            # Use retry wrapper
            response = await self._invoke_with_retry(llm, prompt)
            # Strip <think> tags for gpt-oss-120b or other reasoning models
            content = response.content
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            return content
        except Exception as e:
            logger.error(f"Reduce phase failed: {e}")
            return f"Error gathering final response: {e}"

    async def process_large_content(
        self, 
        content_id: str, 
        task_type: ProcessingTaskType, 
        query: Optional[str] = None,
        strategy: ProcessingStrategy = ProcessingStrategy.STANDARD
    ) -> ProcessingResult:
        """
        Main entry point for intelligent content processing.
        Auto-balances processing based on content size and strategy.
        """
        start_time = time.time()
        
        # 1. Retrieve Content
        metadata, content = self.get_content(content_id)
        if not content:
            raise ValueError(f"Content {content_id} not found")

        # --- OPTIMIZATION: IDEMPOTENCY CHECK ---
        # If we have already optimized this content (created archives), return them immediately.
        # This prevents re-running expensive Map-Reduce on every query.
        if strategy == ProcessingStrategy.CONTEXT_OPTIMIZATION:
            existing_archives = [
                m.id for m in self._registry.values() 
                if f"parent:{content_id}" in m.tags and "archive" in m.tags
            ]
            if existing_archives:
                logger.info(f"⚡ [CACHE HIT] Content {content_id} already has {len(existing_archives)} archives. Skipping Map-Reduce.")
                return ProcessingResult(
                    task_type=task_type,
                    final_output="[Cached] Content already optimized and available for retrieval.",
                    chunk_count=len(existing_archives),
                    processing_time_ms=(time.time() - start_time) * 1000,
                    model_map="cached",
                    model_reduce="cached",
                    metadata={"archived_chunks": existing_archives, "strategy": strategy.value, "cached": True}
                )

        # 2. Chunking (Type-Aware)
        chunks = self._chunk_content(content, metadata.content_type, metadata.name)
        chunk_count = len(chunks)
        logger.info(f"Processing content {content_id} ({metadata.content_type}): {chunk_count} chunks")
        
        # --- STRATEGY: ARCHIVAL_MEMORY ---
        # Special handling for conversation lists
        if strategy == ProcessingStrategy.ARCHIVAL_MEMORY:
            if not isinstance(content, list):
                # Try to parse if it's a string representation of a list
                try: 
                    content_list = json.loads(str(content))
                    if isinstance(content_list, list):
                        content = content_list
                except:
                    pass
            
            if isinstance(content, list):
                # Logic: Keep last N items active, archive the rest
                KEEP_LAST_N = 10
                if len(content) > KEEP_LAST_N:
                    to_archive = content[:-KEEP_LAST_N]
                    active_context = content[-KEEP_LAST_N:]
                    
                    # 1. Generate Summary FIRST (so we can save it in metadata)
                    archive_text = json.dumps(to_archive, default=str)
                    summary_prompt = f"Summarize this conversation segment concisely for context retention:\n{archive_text[:15000]}"
                    
                    archive_summary = "Archived conversation segment."
                    if self._get_llm_client(self.MODEL_HELPER):
                        try:
                            llm_client = self._get_llm_client(self.MODEL_HELPER)
                            summary_res = await self._invoke_with_retry(llm_client, summary_prompt)
                            archive_summary = summary_res.content
                        except Exception as e:
                            logger.warn(f"Summary generation failed: {e}")
                            
                    # 2. Archive "stale" messages with specific summary
                    archive_name = f"archive_{metadata.name}_{int(time.time())}.json"
                    archive_meta = await self.register_content(
                        content=to_archive,
                        name=archive_name,
                        source=ContentSource.SYSTEM_GENERATED,
                        content_type=ContentType.CONVERSATION,
                        priority=ContentPriority.MEDIUM,
                        tags=["archive", f"parent:{content_id}"],
                        thread_id=metadata.thread_id,
                        summary=archive_summary  # Pass the rich summary here!
                    )

                    return ProcessingResult(
                        task_type=task_type,
                        final_output=archive_summary,
                        chunk_count=1,
                        processing_time_ms=(time.time() - start_time) * 1000,
                        model_map=self.MODEL_HELPER,
                        model_reduce=self.MODEL_GENERATOR,
                        metadata={
                            "archived_file_id": archive_meta.id,
                            "archived_count": len(to_archive),
                            "active_count": len(active_context),
                            "strategy": strategy.value
                        }
                    )
                else:
                    return ProcessingResult(
                        task_type=task_type,
                        final_output="No archiving needed (content within limits).",
                        chunk_count=1,
                        processing_time_ms=(time.time() - start_time) * 1000,
                        model_map="none",
                        model_reduce="none",
                        metadata={"strategy": strategy.value, "status": "no_op"}
                    )

        # STRATEGY: CONTEXT_OPTIMIZATION
        # If optimization is requested, we process chunks in parallel to:
        # 1. Generate rich summaries (Map)
        # 2. Register them as retrievable archives
        # 3. Use summaries to build Master Summary (Reduce)
        archived_chunk_ids = []
        
        if strategy == ProcessingStrategy.CONTEXT_OPTIMIZATION:
            chunk_summaries = []
            
            async def process_and_register_chunk(i, chunk):
                # 1. Generate Summary
                chunk_summary = await self._process_chunk_map(chunk, ProcessingTaskType.SUMMARIZE)
                
                # 2. Register with "archive" tag and Summary
                chunk_name = f"{metadata.name}_part_{i+1}.txt"
                chunk_meta = await self.register_content(
                    content=chunk,
                    name=chunk_name,
                    source=ContentSource.SYSTEM_GENERATED,
                    content_type=ContentType.DOCUMENT,
                    priority=ContentPriority.LOW,
                    tags=["archive", "doc_chunk", f"parent:{content_id}"], # 'archive' tag enables retrieval
                    thread_id=metadata.thread_id,
                    summary=chunk_summary
                )
                return chunk_meta.id, chunk_summary

            # Execute in parallel
            logger.info(f"Optimizing context: Processing {len(chunks)} chunks in parallel...")
            tasks = [process_and_register_chunk(i, c) for i, c in enumerate(chunks)]
            results = await asyncio.gather(*tasks)
            
            # Unpack results
            for cid, csum in results:
                archived_chunk_ids.append(cid)
                chunk_summaries.append(csum)
                
            # Reduce Phase: Master Summary
            final_output = await self._process_reduce(chunk_summaries, task_type, query)
            final_output += f"\n\n[System Note: Content was split into {len(archived_chunk_ids)} retrievable archives.]"
            
            return ProcessingResult(
                task_type=task_type,
                final_output=final_output,
                chunk_count=chunk_count,
                processing_time_ms=(time.time() - start_time) * 1000,
                model_map=self.MODEL_HELPER,
                model_reduce=self.MODEL_GENERATOR,
                metadata={"archived_chunks": archived_chunk_ids, "strategy": strategy.value}
            )

        # 3. Map Phase (Parallel) - Standard Strategy Fallback
        map_tasks = [self._process_chunk_map(chunk, task_type, query) for chunk in chunks]
        map_results = await asyncio.gather(*map_tasks)

        # 4. Reduce Phase
        final_output = await self._process_reduce(map_results, task_type, query)
        
        # STRATEGY: DATA_EXTRACTION
        # If the output looks structured (CSV/JSON), save it as a new artifact
        extracted_artifact_id = None
        if strategy == ProcessingStrategy.DATA_EXTRACTION:
            # Simple heuristic detection
            is_json = final_output.strip().startswith("{") or final_output.strip().startswith("[")
            is_csv = "," in final_output and "\n" in final_output
            
            ext = ".txt"
            mime = "text/plain"
            if is_json: 
                ext = ".json"
                mime = "application/json"
            elif is_csv: 
                ext = ".csv"
                mime = "text/csv"
            
            if is_json or is_csv:
                extract_name = f"extracted_{metadata.name}{ext}"
                extract_meta = await self.register_content(
                    content=final_output,
                    name=extract_name,
                    source=ContentSource.SYSTEM_GENERATED,
                    content_type=ContentType.SPREADSHEET if is_csv else ContentType.DOCUMENT,
                    priority=ContentPriority.HIGH, # Extracted data is valuable
                    tags=["extracted", f"parent:{content_id}"],
                    thread_id=metadata.thread_id,
                    mime_type=mime,
                    summary=f"Data extracted from {metadata.name}"
                )
                extracted_artifact_id = extract_meta.id
                final_output += f"\n\n[System Note: Extracted data saved as artifact {extract_name} (ID: {extract_meta.id})]"

        meta_response = {
            "archived_chunks": archived_chunk_ids, 
            "strategy": strategy.value
        }
        if extracted_artifact_id:
            meta_response["extracted_artifact_id"] = extracted_artifact_id
        
        return ProcessingResult(
            task_type=task_type,
            final_output=final_output,
            chunk_count=chunk_count,
            processing_time_ms=(time.time() - start_time) * 1000,
            model_map=self.MODEL_HELPER,
            model_reduce=self.MODEL_GENERATOR,
            metadata=meta_response
        )

    async def retrieve_relevant_context(self, query: str, thread_id: str) -> str:
        """
        Retrieves and synthesizes information from archived conversation history 
        based on the user's query.
        
        Process:
        1. Identify candidate archives for the thread.
        2. [Selector] Helper LLM picks relevant archives based on metadata/summaries.
        3. [Synthesis] Generator LLM reads full content of picked archives and answers query.
        """
        # 1. Identify Candidates (Converation Archives OR Document Chunks)
        candidates = [
            m for m in self.get_by_thread(thread_id) 
            if "archive" in m.tags
        ]
        
        if not candidates:
            return "No archived context found."
            
        logger.info(f"Retrieval: Found {len(candidates)} candidates for thread {thread_id}")
        
        # 2. Selector Phase (Filter by Metadata)
        # Create a lightweight "menu" for the LLM
        menu = "\n".join([
            f"ID: {m.id} | Date: {m.created_at} | Summary: {m.summary or 'No summary'}"
            for m in candidates
        ])
        
        selector_prompt = f"""You are a Context Retrieval System.
User Query: "{query}"

Available Conversation Archives:
{menu}

Task: Identify which Archive IDs might contain information relevant to the User Query.
Return ONLY a JSON list of IDs. Example: ["id1", "id2"]. If none seem relevant, return []."""

        llm_helper = self._get_llm_client(self.MODEL_HELPER)
        if not llm_helper:
            return "Error: Retrieval Service Unavailable (LLM)"
            
        selected_ids = []
        try:
            response = await llm_helper.ainvoke(selector_prompt)
            # Rough parsing of list from text
            text = response.content.strip()
            if "[" in text and "]" in text:
                start = text.find("[")
                end = text.rfind("]") + 1
                import json
                selected_ids = json.loads(text[start:end])
        except Exception as e:
            logger.error(f"Retrieval selection failed: {e}")
            
        if not selected_ids:
            return "No relevant archived context found for your query."
            
        # 3. Retrieval & Synthesis Phase
        logger.info(f"Retrieval: Selected {len(selected_ids)} archives: {selected_ids}")
        context_blocks = []
        for cid in selected_ids:
            # Verify ID belongs to candidates to prevent hallucinated IDs
            if any(c.id == cid for c in candidates):
                _, content = self.get_content(cid)
                if content:
                    context_blocks.append(f"--- Archive {cid} ---\n{str(content)}")
        
        full_context = "\n".join(context_blocks)
        
        synthesis_prompt = f"""You are answering a question based on retrieved conversation history.
        
User Query: "{query}"

Retrieved History:
{full_context[:50000]} #(Limit context)

Task: Answer the user's query using ONLY the retrieved history. If the answer isn't there, say so."""

        llm_gen = self._get_llm_client(self.MODEL_GENERATOR) or llm_helper
        try:
            final_res = await llm_gen.ainvoke(synthesis_prompt)
            return final_res.content
        except Exception as e:
            logger.error(f"Retrieval synthesis failed: {e}")
            return f"Error retrieving context: {e}"
            return f"Error retrieving context: {e}"

    async def cleanup_expired_content(self) -> Dict[str, Any]:
        """
        Garbage Collection: Deletes expired content from disk and registry.
        """
        now = time.time()
        expired_ids = []
        bytes_freed = 0
        
        with self._lock:
            # Identify candidates
            for cid, meta in self._registry.items():
                if meta.expires_at and meta.expires_at < now:
                    expired_ids.append(cid)
            
            # Delete files and update registry
            for cid in expired_ids:
                meta = self._registry[cid]
                file_path = Path(meta.storage_path)
                
                # 1. Delete physical file
                if file_path.exists():
                    try:
                        bytes_freed += file_path.stat().st_size
                        os.remove(file_path)
                    except Exception as e:
                        logger.error(f"Failed to delete expired file {file_path}: {e}")
                
                # 2. Remove from registry
                del self._registry[cid]
            
            if expired_ids:
                self._save_registry()
                logger.info(f"🧹 GC Cleanup: Removed {len(expired_ids)} files, freed {bytes_freed/1024:.2f} KB")
        
        return {
            "deleted_count": len(expired_ids),
            "bytes_freed": bytes_freed,
            "expired_ids": expired_ids
        }

    # =========================================================================
    # CONTENT REGISTRATION & RETRIEVAL (Inherited logic)
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
        with self._lock:
            # Prepare content bytes early for checksum
            if isinstance(content, bytes): content_bytes = content
            elif isinstance(content, str): content_bytes = content.encode('utf-8')
            else: content_bytes = json.dumps(content, default=str, ensure_ascii=False).encode('utf-8')
            
            # --- RE-DO LOGIC CORRECTLY ---
            # 1. Compress if needed
            original_size = len(content_bytes)
            is_compressed = original_size > self.COMPRESSION_THRESHOLD
            final_bytes = gzip.compress(content_bytes) if is_compressed else content_bytes
            
            # 2. Compute Checksum of FINAL (possibly compressed) bytes
            final_checksum = self._compute_checksum(final_bytes)
            
            # 3. Dedupe Check
            if not is_artifact:
                 for meta in self._registry.values():
                    # Also check original_size and compression status to be sure
                    if (meta.checksum == final_checksum and 
                        meta.size_bytes == len(final_bytes) and
                        meta.name == name):
                         logger.info(f"♻️ Content deduplicated: {name} -> Existing ID {meta.id}")
                         if tags:
                            meta.tags = list(set(meta.tags) | set(tags))
                         meta.accessed_at = datetime.utcnow().isoformat()
                         self._save_registry()
                         return meta
            
            # 4. New Content Creation
            content_id = self._generate_id()
            if not mime_type:
                mime_type, _ = mimetypes.guess_type(name)
                mime_type = mime_type or 'application/octet-stream'
            if not content_type:
                content_type = self._determine_content_type(name, mime_type)
            
            ext = Path(name).suffix or '.bin'
            if is_compressed: ext += '.gz'
            storage_path = self._get_storage_path(content_id, content_type, ext)
            
            with open(storage_path, 'wb') as f: f.write(final_bytes)
            
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

    async def register_user_upload(self, file_content: bytes, filename: str, user_id: str, thread_id: Optional[str] = None, mime_type: Optional[str] = None) -> UnifiedContentMetadata:
        return await self.register_content(content=file_content, name=filename, source=ContentSource.USER_UPLOAD, user_id=user_id, thread_id=thread_id, mime_type=mime_type, priority=ContentPriority.HIGH)

    def get_content(self, content_id: str, update_access: bool = True) -> Optional[Tuple[UnifiedContentMetadata, Any]]:
        with self._lock:
            if content_id not in self._registry: return None
            metadata = self._registry[content_id]
            if metadata.expires_at and datetime.utcnow() > datetime.fromisoformat(metadata.expires_at):
                self.delete_content(content_id)
                return None
            
            try:
                with open(metadata.storage_path, 'rb') as f: content_bytes = f.read()
                if metadata.is_compressed: content_bytes = gzip.decompress(content_bytes)
                
                if metadata.is_artifact or metadata.content_type in [ContentType.RESULT, ContentType.PLAN, ContentType.DATA]:
                    try: content = json.loads(content_bytes.decode('utf-8'))
                    except: content = content_bytes.decode('utf-8')
                elif metadata.content_type in [ContentType.IMAGE, ContentType.ARCHIVE]:
                    content = content_bytes
                else:
                    try: content = content_bytes.decode('utf-8')
                    except: content = content_bytes
                
                if update_access:
                    metadata.accessed_at = datetime.utcnow().isoformat()
                    metadata.access_count += 1
                    self._save_registry()
                return metadata, content
            except Exception as e:
                logger.error(f"Failed to retrieve content {content_id}: {e}")
                return None

    def get_metadata(self, content_id: str) -> Optional[UnifiedContentMetadata]:
        return self._registry.get(content_id)

    def get_content_bytes(self, content_id: str) -> Optional[bytes]:
        result = self.get_content(content_id)
        if not result: return None
        metadata, content = result
        if isinstance(content, bytes): return content
        if isinstance(content, str): return content.encode('utf-8')
        return json.dumps(content, default=str).encode('utf-8')

    def get_by_thread(self, thread_id: str) -> List[UnifiedContentMetadata]:
        return [m for m in self._registry.values() if m.thread_id == thread_id]
    
    def get_by_user(self, user_id: str) -> List[UnifiedContentMetadata]:
        return [m for m in self._registry.values() if m.user_id == user_id]

    def delete_content(self, content_id: str) -> bool:
        with self._lock:
            if content_id not in self._registry: return False
            metadata = self._registry[content_id]
            try:
                if os.path.exists(metadata.storage_path): os.remove(metadata.storage_path)
            except Exception as e: logger.error(f"Failed to delete file {content_id}: {e}")
            del self._registry[content_id]
            self._save_registry()
            return True

    def cleanup_expired(self) -> int:
        now = datetime.utcnow()
        expired_ids = [cid for cid, m in self._registry.items() if m.expires_at and now > datetime.fromisoformat(m.expires_at)]
        for cid in expired_ids: self.delete_content(cid)
        return len(expired_ids)
