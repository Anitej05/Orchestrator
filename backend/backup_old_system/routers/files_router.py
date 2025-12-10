"""
File Management API Router

Provides endpoints for:
- File uploads
- File downloads
- File metadata
- File listing
- File deletion
"""

import os
import logging
from typing import List, Optional
from fastapi import APIRouter, File, UploadFile, HTTPException, Query, Depends
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from datetime import datetime

from services.file_management_service import (
    file_manager,
    ManagedFile,
    FileCategory,
    FileSource
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/files", tags=["Files"])


# Response models
class FileMetadataResponse(BaseModel):
    file_id: str
    original_name: str
    file_category: str
    file_source: str
    mime_type: str
    file_size: int
    created_at: str
    download_url: str
    thread_id: Optional[str] = None
    content_summary: Optional[str] = None


class FileUploadResponse(BaseModel):
    success: bool
    files: List[FileMetadataResponse]
    message: str


class FileListResponse(BaseModel):
    files: List[FileMetadataResponse]
    total: int


def managed_file_to_response(mf: ManagedFile) -> FileMetadataResponse:
    """Convert ManagedFile to API response"""
    return FileMetadataResponse(
        file_id=mf.file_id,
        original_name=mf.original_name,
        file_category=mf.file_category.value,
        file_source=mf.file_source.value,
        mime_type=mf.mime_type,
        file_size=mf.file_size,
        created_at=mf.created_at,
        download_url=f"/api/files/download/{mf.file_id}",
        thread_id=mf.thread_id,
        content_summary=mf.content_summary
    )


@router.post("/upload", response_model=FileUploadResponse)
async def upload_files(
    files: List[UploadFile] = File(...),
    user_id: str = Query(default="anonymous"),
    thread_id: Optional[str] = Query(default=None)
):
    """
    Upload one or more files.
    
    Files are stored and tracked by the file management system.
    Returns metadata including file_id for future reference.
    """
    uploaded_files = []
    
    for file in files:
        if not file.filename:
            continue
        
        try:
            content = await file.read()
            managed_file = await file_manager.register_user_upload(
                file_content=content,
                filename=file.filename,
                user_id=user_id,
                thread_id=thread_id,
                mime_type=file.content_type
            )
            uploaded_files.append(managed_file_to_response(managed_file))
        except Exception as e:
            logger.error(f"Failed to upload file {file.filename}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to upload {file.filename}: {str(e)}")
    
    return FileUploadResponse(
        success=True,
        files=uploaded_files,
        message=f"Successfully uploaded {len(uploaded_files)} file(s)"
    )


@router.get("/download/{file_id}")
async def download_file(file_id: str):
    """
    Download a file by its ID.
    
    Returns the file with appropriate content-type headers.
    """
    managed_file = file_manager.get_file(file_id)
    
    if not managed_file:
        raise HTTPException(status_code=404, detail="File not found")
    
    if not os.path.exists(managed_file.stored_path):
        raise HTTPException(status_code=404, detail="File no longer exists on disk")
    
    return FileResponse(
        path=managed_file.stored_path,
        filename=managed_file.original_name,
        media_type=managed_file.mime_type
    )


@router.get("/metadata/{file_id}", response_model=FileMetadataResponse)
async def get_file_metadata(file_id: str):
    """
    Get metadata for a specific file.
    """
    managed_file = file_manager.get_file(file_id)
    
    if not managed_file:
        raise HTTPException(status_code=404, detail="File not found")
    
    return managed_file_to_response(managed_file)


@router.get("/list", response_model=FileListResponse)
async def list_files(
    user_id: Optional[str] = Query(default=None),
    thread_id: Optional[str] = Query(default=None),
    category: Optional[str] = Query(default=None),
    source: Optional[str] = Query(default=None),
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0)
):
    """
    List files with optional filtering.
    
    Filters:
    - user_id: Filter by owner
    - thread_id: Filter by conversation thread
    - category: Filter by file category (image, document, spreadsheet, etc.)
    - source: Filter by source (user_upload, agent_output)
    """
    all_files = list(file_manager._file_registry.values())
    
    # Apply filters
    if user_id:
        all_files = [f for f in all_files if f.user_id == user_id]
    
    if thread_id:
        all_files = [f for f in all_files if f.thread_id == thread_id]
    
    if category:
        try:
            cat = FileCategory(category)
            all_files = [f for f in all_files if f.file_category == cat]
        except ValueError:
            pass
    
    if source:
        try:
            src = FileSource(source)
            all_files = [f for f in all_files if f.file_source == src]
        except ValueError:
            pass
    
    # Sort by created_at descending
    all_files.sort(key=lambda f: f.created_at, reverse=True)
    
    # Paginate
    total = len(all_files)
    paginated = all_files[offset:offset + limit]
    
    return FileListResponse(
        files=[managed_file_to_response(f) for f in paginated],
        total=total
    )


@router.get("/thread/{thread_id}", response_model=FileListResponse)
async def get_thread_files(thread_id: str):
    """
    Get all files associated with a conversation thread.
    """
    files = file_manager.get_files_for_thread(thread_id)
    
    return FileListResponse(
        files=[managed_file_to_response(f) for f in files],
        total=len(files)
    )


@router.delete("/{file_id}")
async def delete_file(file_id: str):
    """
    Delete a file by its ID.
    """
    success = file_manager.delete_file(file_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="File not found")
    
    return {"success": True, "message": f"File {file_id} deleted"}


@router.post("/cleanup")
async def cleanup_expired():
    """
    Clean up expired temporary files.
    
    This is typically called by a scheduled job.
    """
    count = file_manager.cleanup_expired_files()
    return {"success": True, "cleaned_up": count}


@router.get("/agent-mapping/{file_id}/{agent_id}")
async def get_agent_file_id(file_id: str, agent_id: str):
    """
    Get the agent-specific file ID for a managed file.
    
    This is useful for checking if a file has been uploaded to a specific agent.
    """
    agent_file_id = file_manager.get_agent_file_id(file_id, agent_id)
    
    if not agent_file_id:
        raise HTTPException(
            status_code=404, 
            detail=f"No mapping found for file {file_id} on agent {agent_id}"
        )
    
    return {
        "orchestrator_file_id": file_id,
        "agent_id": agent_id,
        "agent_file_id": agent_file_id
    }
