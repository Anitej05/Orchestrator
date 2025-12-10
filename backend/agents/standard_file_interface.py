"""
Standard Agent File Interface

This module provides a standard file handling interface that agents can use
to implement consistent file management across all agents.

Usage:
    from standard_file_interface import StandardFileHandler, create_file_endpoints

    # In your agent:
    file_handler = StandardFileHandler(storage_dir="storage/my_agent")
    
    # Add standard endpoints to your FastAPI app
    create_file_endpoints(app, file_handler)
"""

import os
import uuid
import logging
import hashlib
import mimetypes
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class AgentFileMetadata:
    """Standard file metadata for agents"""
    file_id: str
    name: str
    size: int
    type: str
    mime_type: str
    checksum: str
    storage_path: str
    created_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FileUploadResponse(BaseModel):
    """Standard response for file uploads"""
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class FileMetadataResponse(BaseModel):
    """Standard response for file metadata"""
    file_id: str
    name: str
    size: int
    type: str
    mime_type: str
    created_at: str


# =============================================================================
# STANDARD FILE HANDLER
# =============================================================================

class StandardFileHandler:
    """
    Standard file handler that agents can use for consistent file management.
    
    Features:
    - Unique file ID generation
    - Checksum computation
    - Metadata tracking
    - File verification
    - Cleanup utilities
    """
    
    def __init__(self, storage_dir: str = "storage/agent_files"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory file registry
        self._files: Dict[str, AgentFileMetadata] = {}
        
        logger.info(f"StandardFileHandler initialized at {self.storage_dir}")
    
    def _generate_file_id(self) -> str:
        """Generate unique file ID"""
        return str(uuid.uuid4())
    
    def _compute_checksum(self, content: bytes) -> str:
        """Compute SHA256 checksum"""
        return hashlib.sha256(content).hexdigest()
    
    def _determine_file_type(self, filename: str, mime_type: Optional[str]) -> str:
        """Determine file type from filename and mime type"""
        ext = Path(filename).suffix.lower()
        
        if ext in ['.csv', '.xlsx', '.xls']:
            return 'spreadsheet'
        elif ext in ['.pdf', '.doc', '.docx', '.txt']:
            return 'document'
        elif mime_type and mime_type.startswith('image/'):
            return 'image'
        elif ext in ['.json', '.xml', '.yaml']:
            return 'data'
        else:
            return 'other'
    
    async def upload_file(
        self,
        file_content: bytes,
        filename: str,
        mime_type: Optional[str] = None
    ) -> AgentFileMetadata:
        """
        Upload and register a file.
        
        Args:
            file_content: Raw file bytes
            filename: Original filename
            mime_type: Optional MIME type
        
        Returns:
            AgentFileMetadata for the uploaded file
        """
        file_id = self._generate_file_id()
        
        # Determine MIME type
        if not mime_type:
            mime_type, _ = mimetypes.guess_type(filename)
            mime_type = mime_type or 'application/octet-stream'
        
        # Determine file type
        file_type = self._determine_file_type(filename, mime_type)
        
        # Create storage path
        ext = Path(filename).suffix
        storage_path = self.storage_dir / f"{file_id}{ext}"
        
        # Write file
        with open(storage_path, 'wb') as f:
            f.write(file_content)
        
        # Compute checksum
        checksum = self._compute_checksum(file_content)
        
        # Create metadata
        metadata = AgentFileMetadata(
            file_id=file_id,
            name=filename,
            size=len(file_content),
            type=file_type,
            mime_type=mime_type,
            checksum=checksum,
            storage_path=str(storage_path),
            created_at=datetime.utcnow().isoformat()
        )
        
        # Register
        self._files[file_id] = metadata
        
        logger.info(f"Uploaded file: {filename} -> {file_id}")
        return metadata
    
    def get_file_metadata(self, file_id: str) -> Optional[AgentFileMetadata]:
        """Get metadata for a file"""
        return self._files.get(file_id)
    
    def get_file_path(self, file_id: str) -> Optional[str]:
        """Get storage path for a file"""
        metadata = self._files.get(file_id)
        return metadata.storage_path if metadata else None
    
    def get_file_content(self, file_id: str) -> Optional[bytes]:
        """Read and return file content"""
        metadata = self._files.get(file_id)
        if not metadata:
            return None
        
        try:
            with open(metadata.storage_path, 'rb') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read file {file_id}: {e}")
            return None
    
    def verify_file(self, file_id: str) -> bool:
        """Verify that a file exists and is accessible"""
        metadata = self._files.get(file_id)
        if not metadata:
            return False
        
        return os.path.exists(metadata.storage_path)
    
    def delete_file(self, file_id: str) -> bool:
        """Delete a file"""
        metadata = self._files.get(file_id)
        if not metadata:
            return False
        
        try:
            if os.path.exists(metadata.storage_path):
                os.remove(metadata.storage_path)
            del self._files[file_id]
            logger.info(f"Deleted file: {file_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete file {file_id}: {e}")
            return False
    
    def list_files(self) -> List[AgentFileMetadata]:
        """List all files"""
        return list(self._files.values())
    
    def cleanup_old_files(self, max_age_hours: int = 24) -> int:
        """Remove files older than max_age_hours"""
        from datetime import timedelta
        
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        old_files = []
        
        for file_id, metadata in self._files.items():
            created_at = datetime.fromisoformat(metadata.created_at)
            if created_at < cutoff:
                old_files.append(file_id)
        
        for file_id in old_files:
            self.delete_file(file_id)
        
        return len(old_files)


# =============================================================================
# FASTAPI ENDPOINT FACTORY
# =============================================================================

def create_file_endpoints(app: FastAPI, handler: StandardFileHandler, prefix: str = ""):
    """
    Add standard file management endpoints to a FastAPI app.
    
    Endpoints created:
    - POST /upload - Upload a file
    - GET /files/{file_id} - Get file metadata
    - GET /files/{file_id}/download - Download file
    - DELETE /files/{file_id} - Delete file
    - GET /files - List all files
    - POST /files/{file_id}/verify - Verify file exists
    
    Args:
        app: FastAPI application
        handler: StandardFileHandler instance
        prefix: Optional URL prefix
    """
    
    @app.post(f"{prefix}/upload", response_model=FileUploadResponse)
    async def upload_file(file: UploadFile = File(...)):
        """Upload a file and get its file_id"""
        try:
            if not file.filename:
                raise HTTPException(status_code=400, detail="Filename required")
            
            content = await file.read()
            metadata = await handler.upload_file(
                file_content=content,
                filename=file.filename,
                mime_type=file.content_type
            )
            
            return FileUploadResponse(
                success=True,
                result={
                    "file_id": metadata.file_id,
                    "name": metadata.name,
                    "size": metadata.size,
                    "type": metadata.type
                }
            )
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return FileUploadResponse(success=False, error=str(e))
    
    @app.get(f"{prefix}/files/{{file_id}}", response_model=FileMetadataResponse)
    async def get_file_metadata(file_id: str):
        """Get metadata for a file"""
        metadata = handler.get_file_metadata(file_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileMetadataResponse(
            file_id=metadata.file_id,
            name=metadata.name,
            size=metadata.size,
            type=metadata.type,
            mime_type=metadata.mime_type,
            created_at=metadata.created_at
        )
    
    @app.get(f"{prefix}/files/{{file_id}}/download")
    async def download_file(file_id: str):
        """Download a file"""
        from fastapi.responses import FileResponse
        
        metadata = handler.get_file_metadata(file_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="File not found")
        
        if not os.path.exists(metadata.storage_path):
            raise HTTPException(status_code=404, detail="File no longer exists")
        
        return FileResponse(
            path=metadata.storage_path,
            filename=metadata.name,
            media_type=metadata.mime_type
        )
    
    @app.delete(f"{prefix}/files/{{file_id}}")
    async def delete_file(file_id: str):
        """Delete a file"""
        if handler.delete_file(file_id):
            return {"success": True, "message": f"File {file_id} deleted"}
        else:
            raise HTTPException(status_code=404, detail="File not found")
    
    @app.get(f"{prefix}/files")
    async def list_files():
        """List all files"""
        files = handler.list_files()
        return {
            "files": [
                {
                    "file_id": f.file_id,
                    "name": f.name,
                    "size": f.size,
                    "type": f.type,
                    "created_at": f.created_at
                }
                for f in files
            ],
            "total": len(files)
        }
    
    @app.post(f"{prefix}/files/{{file_id}}/verify")
    async def verify_file(file_id: str):
        """Verify that a file exists"""
        exists = handler.verify_file(file_id)
        return {"file_id": file_id, "exists": exists}
    
    logger.info(f"Added standard file endpoints with prefix '{prefix}'")


# =============================================================================
# CONVENIENCE FUNCTION FOR QUICK SETUP
# =============================================================================

def setup_standard_file_handling(
    app: FastAPI,
    agent_name: str,
    storage_subdir: Optional[str] = None
) -> StandardFileHandler:
    """
    Quick setup for standard file handling in an agent.
    
    Args:
        app: FastAPI application
        agent_name: Name of the agent (used for storage directory)
        storage_subdir: Optional custom storage subdirectory
    
    Returns:
        StandardFileHandler instance
    
    Example:
        app = FastAPI(title="My Agent")
        file_handler = setup_standard_file_handling(app, "my_agent")
    """
    storage_dir = storage_subdir or f"storage/{agent_name}"
    handler = StandardFileHandler(storage_dir=storage_dir)
    create_file_endpoints(app, handler)
    return handler
