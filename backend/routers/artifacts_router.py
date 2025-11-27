"""
Artifacts Router - API endpoints for artifact management

Provides REST API for:
- Storing and retrieving artifacts
- Listing artifacts by thread/type
- Managing artifact lifecycle
- Getting optimized context
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime

from services.artifact_service import (
    ArtifactStore,
    ArtifactType,
    ArtifactPriority,
    ArtifactMetadata,
    Artifact,
    ArtifactReference,
    get_artifact_store,
    get_context_manager,
    get_state_manager
)
from services.context_optimizer import (
    ContextOptimizer,
    OptimizedContext,
    get_context_optimizer
)

router = APIRouter(prefix="/api/artifacts", tags=["artifacts"])


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class StoreArtifactRequest(BaseModel):
    """Request to store a new artifact"""
    content: Any = Field(..., description="The content to store")
    name: str = Field(..., description="Human-readable name")
    type: Literal["code", "document", "data", "result", "canvas", "screenshot", "plan", "conversation", "embedding", "summary"]
    description: str = Field("", description="Description of the artifact")
    thread_id: Optional[str] = Field(None, description="Associated conversation thread")
    priority: Literal["critical", "high", "medium", "low", "ephemeral"] = "medium"
    tags: List[str] = Field(default_factory=list, description="Searchable tags")
    ttl_hours: Optional[int] = Field(None, description="Time-to-live in hours")


class ArtifactMetadataResponse(BaseModel):
    """Response containing artifact metadata"""
    id: str
    type: str
    name: str
    description: str
    priority: str
    thread_id: Optional[str]
    tags: List[str]
    size_bytes: int
    compressed: bool
    created_at: str
    updated_at: str
    accessed_at: str
    access_count: int
    summary: Optional[str]


class ArtifactResponse(BaseModel):
    """Response containing full artifact"""
    metadata: ArtifactMetadataResponse
    content: Any


class ArtifactReferenceResponse(BaseModel):
    """Lightweight artifact reference"""
    id: str
    type: str
    name: str
    summary: str
    size_bytes: int


class ThreadContextResponse(BaseModel):
    """Optimized context for a thread"""
    references: List[ArtifactReferenceResponse]
    context_string: str
    total_tokens_saved: int
    artifact_count: int


class OptimizedContextRequest(BaseModel):
    """Request for optimized context"""
    state: Dict[str, Any] = Field(..., description="The orchestrator state")
    thread_id: str = Field(..., description="Conversation thread ID")
    max_tokens: int = Field(8000, description="Maximum tokens for context")
    focus_fields: List[str] = Field(default_factory=list, description="Fields to prioritize")


class OptimizedContextResponse(BaseModel):
    """Response with optimized context"""
    user_context: str
    artifact_references: List[Dict[str, Any]]
    total_tokens: int
    tokens_saved: int
    compression_ratio: float
    included_count: int
    excluded_count: int


class StorageStatsResponse(BaseModel):
    """Storage statistics"""
    total_artifacts: int
    total_size_bytes: int
    total_size_mb: float
    by_type: Dict[str, Dict[str, Any]]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def metadata_to_response(meta: ArtifactMetadata) -> ArtifactMetadataResponse:
    """Convert ArtifactMetadata to response model"""
    return ArtifactMetadataResponse(
        id=meta.id,
        type=meta.type.value,
        name=meta.name,
        description=meta.description,
        priority=meta.priority.value,
        thread_id=meta.thread_id,
        tags=meta.tags,
        size_bytes=meta.size_bytes,
        compressed=meta.compressed,
        created_at=datetime.fromtimestamp(meta.created_at).isoformat(),
        updated_at=datetime.fromtimestamp(meta.updated_at).isoformat(),
        accessed_at=datetime.fromtimestamp(meta.accessed_at).isoformat(),
        access_count=meta.access_count,
        summary=meta.summary
    )


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/store", response_model=ArtifactMetadataResponse)
async def store_artifact(request: StoreArtifactRequest):
    """
    Store a new artifact.
    
    Large content is automatically compressed. Returns metadata with artifact ID.
    """
    store = get_artifact_store()
    
    try:
        metadata = store.store(
            content=request.content,
            name=request.name,
            artifact_type=ArtifactType(request.type),
            description=request.description,
            thread_id=request.thread_id,
            priority=ArtifactPriority(request.priority),
            tags=request.tags,
            ttl_hours=request.ttl_hours
        )
        return metadata_to_response(metadata)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store artifact: {str(e)}")


@router.get("/{artifact_id}", response_model=ArtifactResponse)
async def get_artifact(artifact_id: str):
    """
    Retrieve an artifact by ID.
    
    Returns full artifact content. Updates access statistics.
    """
    store = get_artifact_store()
    artifact = store.retrieve(artifact_id)
    
    if not artifact:
        raise HTTPException(status_code=404, detail=f"Artifact not found: {artifact_id}")
    
    return ArtifactResponse(
        metadata=metadata_to_response(artifact.metadata),
        content=artifact.content
    )


@router.get("/{artifact_id}/reference", response_model=ArtifactReferenceResponse)
async def get_artifact_reference(artifact_id: str):
    """
    Get a lightweight reference to an artifact.
    
    Returns summary and metadata without full content.
    """
    store = get_artifact_store()
    artifact = store.retrieve(artifact_id, update_access=False)
    
    if not artifact:
        raise HTTPException(status_code=404, detail=f"Artifact not found: {artifact_id}")
    
    ref = artifact.to_reference()
    return ArtifactReferenceResponse(
        id=ref.id,
        type=ref.type.value,
        name=ref.name,
        summary=ref.summary,
        size_bytes=ref.size_bytes
    )


@router.delete("/{artifact_id}")
async def delete_artifact(artifact_id: str):
    """Delete an artifact by ID."""
    store = get_artifact_store()
    
    if store.delete(artifact_id):
        return {"status": "deleted", "artifact_id": artifact_id}
    else:
        raise HTTPException(status_code=404, detail=f"Artifact not found: {artifact_id}")


@router.get("/thread/{thread_id}", response_model=List[ArtifactMetadataResponse])
async def list_thread_artifacts(thread_id: str):
    """List all artifacts for a conversation thread."""
    store = get_artifact_store()
    artifacts = store.list_by_thread(thread_id)
    return [metadata_to_response(meta) for meta in artifacts]


@router.get("/type/{artifact_type}", response_model=List[ArtifactMetadataResponse])
async def list_artifacts_by_type(
    artifact_type: Literal["code", "document", "data", "result", "canvas", "screenshot", "plan", "conversation", "embedding", "summary"]
):
    """List all artifacts of a specific type."""
    store = get_artifact_store()
    artifacts = store.list_by_type(ArtifactType(artifact_type))
    return [metadata_to_response(meta) for meta in artifacts]


@router.get("/search/tags", response_model=List[ArtifactMetadataResponse])
async def search_by_tags(tags: str = Query(..., description="Comma-separated tags")):
    """Search artifacts by tags."""
    store = get_artifact_store()
    tag_list = [t.strip() for t in tags.split(",")]
    artifacts = store.search_by_tags(tag_list)
    return [metadata_to_response(meta) for meta in artifacts]


@router.get("/thread/{thread_id}/context", response_model=ThreadContextResponse)
async def get_thread_context(thread_id: str):
    """
    Get optimized context for a thread.
    
    Returns artifact references with summaries instead of full content.
    """
    context_manager = get_context_manager()
    result = context_manager.get_context_with_references(thread_id)
    
    return ThreadContextResponse(
        references=[
            ArtifactReferenceResponse(
                id=ref.id,
                type=ref.type.value,
                name=ref.name,
                summary=ref.summary,
                size_bytes=ref.size_bytes
            )
            for ref in result["references"]
        ],
        context_string=result["context_string"],
        total_tokens_saved=result["total_tokens_saved"],
        artifact_count=result["artifact_count"]
    )


@router.post("/optimize-context", response_model=OptimizedContextResponse)
async def optimize_context(request: OptimizedContextRequest):
    """
    Optimize orchestrator state for LLM context.
    
    Compresses large fields into artifacts and returns optimized context.
    """
    optimizer = get_context_optimizer()
    
    result = optimizer.optimize_state_for_context(
        state=request.state,
        thread_id=request.thread_id,
        max_tokens=request.max_tokens,
        focus_fields=request.focus_fields
    )
    
    return OptimizedContextResponse(
        user_context=result.user_context,
        artifact_references=result.artifact_references,
        total_tokens=result.total_tokens,
        tokens_saved=result.tokens_saved,
        compression_ratio=result.compression_ratio,
        included_count=len(result.included_blocks),
        excluded_count=len(result.excluded_blocks)
    )


@router.post("/cleanup")
async def cleanup_expired():
    """Remove expired artifacts."""
    store = get_artifact_store()
    count = store.cleanup_expired()
    return {"status": "completed", "removed_count": count}


@router.get("/stats", response_model=StorageStatsResponse)
async def get_storage_stats():
    """Get artifact storage statistics."""
    store = get_artifact_store()
    stats = store.get_stats()
    return StorageStatsResponse(**stats)
