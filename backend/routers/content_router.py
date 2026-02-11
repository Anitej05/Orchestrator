"""
Unified Content Router - API endpoints for content management

Provides REST API for:
- Content uploads (files and artifacts)
- Content retrieval and downloads
- Content metadata and listing
- Agent content mapping
- Lifecycle management
- Statistics
"""

import os
import logging
from typing import List, Optional, Any, Literal
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from datetime import datetime

from backend.services.content_management_service import (
    ContentManagementService,
    UnifiedContentMetadata,
    ContentType,
    ContentSource,
    ContentPriority,
    RetentionPolicy,
    ContentReference,
)

# Helper to get service instance
_content_service = None

def get_content_service():
    global _content_service
    if _content_service is None:
        _content_service = ContentManagementService()
    return _content_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/content", tags=["Content"])


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class ContentMetadataResponse(BaseModel):
    """Response containing content metadata"""
    id: str
    name: str
    content_type: str
    source: str
    mime_type: str
    size_bytes: int
    is_compressed: bool
    is_artifact: bool
    priority: str
    created_at: str
    expires_at: Optional[str]
    summary: Optional[str]
    tags: List[str]
    download_url: str
    thread_id: Optional[str] = None
    access_count: int = 0


class ContentUploadResponse(BaseModel):
    """Response for content upload"""
    success: bool
    content: List[ContentMetadataResponse]
    message: str


class ContentListResponse(BaseModel):
    """Response for content listing"""
    content: List[ContentMetadataResponse]
    total: int


class StoreArtifactRequest(BaseModel):
    """Request to store an artifact"""
    content: Any = Field(..., description="The content to store")
    name: str = Field(..., description="Human-readable name")
    content_type: Literal[
        "image", "document", "spreadsheet", "code", "data", "archive",
        "canvas", "screenshot", "result", "plan", "conversation", "summary", "other"
    ]
    thread_id: Optional[str] = Field(None, description="Associated thread")
    description: Optional[str] = Field(None, description="Description")
    priority: Literal["critical", "high", "medium", "low", "ephemeral"] = "medium"
    tags: List[str] = Field(default_factory=list)
    ttl_hours: Optional[int] = Field(None, description="Time-to-live in hours")


class ContentReferenceResponse(BaseModel):
    """Lightweight content reference"""
    id: str
    name: str
    content_type: str
    summary: str
    size_bytes: int


class AgentMappingResponse(BaseModel):
    """Agent content mapping"""
    content_id: str
    agent_id: str
    agent_content_id: str
    created_at: str
    is_valid: bool


class OptimizedContextRequest(BaseModel):
    """Request for optimized context"""
    thread_id: str
    max_tokens: int = 8000
    include_summaries: bool = True


class OptimizedContextResponse(BaseModel):
    """Response with optimized context"""
    context_string: str
    references: List[ContentReferenceResponse]
    tokens_saved: int
    content_count: int


class StorageStatsResponse(BaseModel):
    """Storage statistics"""
    total_items: int
    total_size_bytes: int
    total_size_mb: float
    by_type: dict
    by_source: dict


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def metadata_to_response(m: UnifiedContentMetadata) -> ContentMetadataResponse:
    """Convert metadata to API response"""
    return ContentMetadataResponse(
        id=m.id,
        name=m.name,
        content_type=m.content_type.value,
        source=m.source.value,
        mime_type=m.mime_type,
        size_bytes=m.size_bytes,
        is_compressed=m.is_compressed,
        is_artifact=m.is_artifact,
        priority=m.priority.value,
        created_at=m.created_at,
        expires_at=m.expires_at,
        summary=m.summary,
        tags=m.tags,
        download_url=f"/api/content/download/{m.id}",
        thread_id=m.thread_id,
        access_count=m.access_count
    )


# =============================================================================
# UPLOAD ENDPOINTS
# =============================================================================

@router.post("/upload", response_model=ContentUploadResponse)
async def upload_files(
    files: List[UploadFile] = File(...),
    user_id: str = Query(default="anonymous"),
    thread_id: Optional[str] = Query(default=None)
):
    """
    Upload one or more files.
    
    Files are stored and tracked by the unified content system.
    Returns metadata including content_id for future reference.
    """
    service = get_content_service()
    uploaded = []
    
    for file in files:
        if not file.filename:
            continue
        
        try:
            content = await file.read()
            metadata = await service.register_user_upload(
                file_content=content,
                filename=file.filename,
                user_id=user_id,
                thread_id=thread_id,
                mime_type=file.content_type
            )
            uploaded.append(metadata_to_response(metadata))
        except Exception as e:
            logger.error(f"Failed to upload {file.filename}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to upload {file.filename}: {str(e)}")
    
    return ContentUploadResponse(
        success=True,
        content=uploaded,
        message=f"Successfully uploaded {len(uploaded)} file(s)"
    )


@router.post("/store", response_model=ContentMetadataResponse)
async def store_artifact(request: StoreArtifactRequest):
    """
    Store an artifact (large content like results, canvas, etc.).
    
    Large content is automatically compressed.
    """
    service = get_content_service()
    
    try:
        metadata = await service.register_artifact(
            content=request.content,
            name=request.name,
            content_type=ContentType(request.content_type),
            thread_id=request.thread_id,
            description=request.description,
            priority=ContentPriority(request.priority),
            ttl_hours=request.ttl_hours
        )
        
        if request.tags:
            metadata.tags = request.tags
            service._save_registry()
        
        return metadata_to_response(metadata)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store artifact: {str(e)}")


# =============================================================================
# RETRIEVAL ENDPOINTS
# =============================================================================

@router.get("/download/{content_id}")
async def download_content(content_id: str):
    """
    Download content by ID.
    
    Returns the file with appropriate content-type headers.
    """
    service = get_content_service()
    metadata = service.get_metadata(content_id)
    
    if not metadata:
        raise HTTPException(status_code=404, detail="Content not found")
    
    if not os.path.exists(metadata.storage_path):
        raise HTTPException(status_code=404, detail="Content file no longer exists")
    
    # For compressed files, we need to decompress
    if metadata.is_compressed:
        content_bytes = service.get_content_bytes(content_id)
        if not content_bytes:
            raise HTTPException(status_code=500, detail="Failed to read content")
        
        return StreamingResponse(
            iter([content_bytes]),
            media_type=metadata.mime_type,
            headers={"Content-Disposition": f"attachment; filename={metadata.name}"}
        )
    
    return FileResponse(
        path=metadata.storage_path,
        filename=metadata.name,
        media_type=metadata.mime_type
    )


@router.get("/{content_id}", response_model=ContentMetadataResponse)
async def get_content_metadata(content_id: str):
    """Get metadata for specific content."""
    service = get_content_service()
    metadata = service.get_metadata(content_id)
    
    if not metadata:
        raise HTTPException(status_code=404, detail="Content not found")
    
    return metadata_to_response(metadata)


@router.get("/{content_id}/reference", response_model=ContentReferenceResponse)
async def get_content_reference(content_id: str):
    """Get a lightweight reference to content."""
    service = get_content_service()
    metadata = service.get_metadata(content_id)
    
    if not metadata:
        raise HTTPException(status_code=404, detail="Content not found")
    
    ref = metadata.to_reference()
    return ContentReferenceResponse(
        id=ref.id,
        name=ref.name,
        content_type=ref.content_type.value,
        summary=ref.summary,
        size_bytes=ref.size_bytes
    )


@router.get("/{content_id}/full")
async def get_full_content(content_id: str):
    """Get full content including data."""
    service = get_content_service()
    result = service.get_content(content_id)
    
    if not result:
        raise HTTPException(status_code=404, detail="Content not found")
    
    metadata, content = result
    
    return {
        "metadata": metadata_to_response(metadata),
        "content": content
    }


# =============================================================================
# LISTING ENDPOINTS
# =============================================================================

@router.get("/list/all", response_model=ContentListResponse)
async def list_all_content(
    user_id: Optional[str] = Query(default=None),
    thread_id: Optional[str] = Query(default=None),
    content_type: Optional[str] = Query(default=None),
    source: Optional[str] = Query(default=None),
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0)
):
    """
    List content with optional filtering.
    """
    service = get_content_service()
    all_content = list(service._registry.values())
    
    # Apply filters
    if user_id:
        all_content = [c for c in all_content if c.user_id == user_id]
    
    if thread_id:
        all_content = [c for c in all_content if c.thread_id == thread_id]
    
    if content_type:
        try:
            ct = ContentType(content_type)
            all_content = [c for c in all_content if c.content_type == ct]
        except ValueError:
            pass
    
    if source:
        try:
            src = ContentSource(source)
            all_content = [c for c in all_content if c.source == src]
        except ValueError:
            pass
    
    # Sort by created_at descending
    all_content.sort(key=lambda c: c.created_at, reverse=True)
    
    # Paginate
    total = len(all_content)
    paginated = all_content[offset:offset + limit]
    
    return ContentListResponse(
        content=[metadata_to_response(c) for c in paginated],
        total=total
    )


@router.get("/thread/{thread_id}", response_model=ContentListResponse)
async def get_thread_content(thread_id: str):
    """Get all content for a thread."""
    service = get_content_service()
    content_list = service.get_by_thread(thread_id)
    
    return ContentListResponse(
        content=[metadata_to_response(c) for c in content_list],
        total=len(content_list)
    )


@router.get("/type/{content_type}", response_model=ContentListResponse)
async def get_content_by_type(
    content_type: Literal[
        "image", "document", "spreadsheet", "code", "data", "archive",
        "canvas", "screenshot", "result", "plan", "conversation", "summary", "other"
    ]
):
    """Get all content of a specific type."""
    service = get_content_service()
    content_list = service.get_by_type(ContentType(content_type))
    
    return ContentListResponse(
        content=[metadata_to_response(c) for c in content_list],
        total=len(content_list)
    )


@router.get("/search/tags", response_model=ContentListResponse)
async def search_by_tags(tags: str = Query(..., description="Comma-separated tags")):
    """Search content by tags."""
    service = get_content_service()
    tag_list = [t.strip() for t in tags.split(",")]
    content_list = service.get_by_tags(tag_list)
    
    return ContentListResponse(
        content=[metadata_to_response(c) for c in content_list],
        total=len(content_list)
    )


# =============================================================================
# AGENT MAPPING ENDPOINTS
# =============================================================================

@router.get("/{content_id}/agent-mapping/{agent_id}", response_model=AgentMappingResponse)
async def get_agent_mapping(content_id: str, agent_id: str):
    """Get the agent-specific content ID for content."""
    service = get_content_service()
    metadata = service.get_metadata(content_id)
    
    if not metadata:
        raise HTTPException(status_code=404, detail="Content not found")
    
    mapping = metadata.agent_mappings.get(agent_id)
    if not mapping:
        raise HTTPException(status_code=404, detail=f"No mapping for agent {agent_id}")
    
    return AgentMappingResponse(
        content_id=mapping.content_id,
        agent_id=mapping.agent_id,
        agent_content_id=mapping.agent_content_id,
        created_at=mapping.created_at,
        is_valid=mapping.is_valid
    )


@router.post("/{content_id}/upload-to-agent/{agent_id}")
async def upload_to_agent(
    content_id: str,
    agent_id: str,
    agent_base_url: str = Query(..., description="Agent's base URL"),
    force: bool = Query(default=False, description="Force re-upload")
):
    """Upload content to an agent and get the agent's content ID."""
    service = get_content_service()
    
    agent_content_id = await service.upload_to_agent(
        content_id=content_id,
        agent_id=agent_id,
        agent_base_url=agent_base_url,
        force_reupload=force
    )
    
    if not agent_content_id:
        raise HTTPException(status_code=500, detail="Failed to upload to agent")
    
    return {
        "content_id": content_id,
        "agent_id": agent_id,
        "agent_content_id": agent_content_id
    }


# =============================================================================
# CONTEXT OPTIMIZATION ENDPOINTS
# =============================================================================

@router.post("/optimize-context", response_model=OptimizedContextResponse)
async def optimize_context(request: OptimizedContextRequest):
    """Get optimized context for a thread."""
    service = get_content_service()
    
    result = service.get_optimized_context(
        thread_id=request.thread_id,
        max_tokens=request.max_tokens,
        include_summaries=request.include_summaries
    )
    
    return OptimizedContextResponse(
        context_string=result["context_string"],
        references=[
            ContentReferenceResponse(**r) for r in result["references"]
        ],
        tokens_saved=result["tokens_saved"],
        content_count=result["content_count"]
    )


# =============================================================================
# LIFECYCLE MANAGEMENT ENDPOINTS
# =============================================================================

@router.delete("/{content_id}")
async def delete_content(content_id: str):
    """Delete content by ID."""
    service = get_content_service()
    
    if service.delete_content(content_id):
        return {"success": True, "message": f"Content {content_id} deleted"}
    else:
        raise HTTPException(status_code=404, detail="Content not found")


@router.post("/cleanup/expired")
async def cleanup_expired():
    """Remove expired content."""
    service = get_content_service()
    count = service.cleanup_expired()
    return {"success": True, "removed_count": count}


@router.post("/cleanup/session/{thread_id}")
async def cleanup_session(thread_id: str):
    """Remove session-only content for a thread."""
    service = get_content_service()
    count = service.cleanup_session(thread_id)
    return {"success": True, "removed_count": count}


# =============================================================================
# STATISTICS ENDPOINT
# =============================================================================

@router.get("/stats/overview", response_model=StorageStatsResponse)
async def get_storage_stats():
    """Get storage statistics."""
    service = get_content_service()
    stats = service.get_stats()
    return StorageStatsResponse(**stats)


# =============================================================================
# BACKWARD COMPATIBILITY - File endpoints
# =============================================================================

# These endpoints maintain backward compatibility with the old file management API

@router.post("/files/upload", response_model=ContentUploadResponse)
async def upload_files_compat(
    files: List[UploadFile] = File(...),
    user_id: str = Query(default="anonymous"),
    thread_id: Optional[str] = Query(default=None)
):
    """Backward compatible file upload endpoint."""
    return await upload_files(files, user_id, thread_id)


@router.get("/files/download/{file_id}")
async def download_file_compat(file_id: str):
    """Backward compatible file download endpoint."""
    return await download_content(file_id)


@router.get("/files/{file_id}", response_model=ContentMetadataResponse)
async def get_file_metadata_compat(file_id: str):
    """Backward compatible file metadata endpoint."""
    return await get_content_metadata(file_id)


# =============================================================================
# FILE PROCESSING CACHE ENDPOINTS
# =============================================================================

@router.get("/cache/stats")
async def get_file_cache_stats():
    """
    Get statistics about the file processing cache.
    
    Returns:
    - cached_files: Number of files in cache
    - cache_keys: List of cache keys (file:hash pairs)
    """
    from backend.services.file_processor import file_processor
    
    stats = file_processor.get_cache_stats()
    return {
        "status": "success",
        "cache_enabled": True,
        "cached_files": stats['cached_files'],
        "cache_keys": stats['cache_keys']
    }


@router.post("/cache/clear")
async def clear_file_cache():
    """
    Clear the file processing cache.
    Use this to force re-processing of all documents.
    """
    from backend.services.file_processor import file_processor
    
    file_processor.clear_cache()
    return {
        "status": "success",
        "message": "File cache cleared"
    }

