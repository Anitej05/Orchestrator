"""
Files Router - Handles file upload and serving endpoints.

Extracted from main.py to improve code organization and maintainability.
"""

import os
from typing import List
from fastapi import APIRouter, HTTPException, File, UploadFile
from fastapi.responses import FileResponse
from aiofiles import open as aio_open
from urllib.parse import unquote
from mimetypes import guess_type

from schemas import FileObject

router = APIRouter(prefix="/api", tags=["Files"])

# Ensure storage directories exist
# Ensure storage directories exist
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
STORAGE_ROOT = PROJECT_ROOT / "storage"
(STORAGE_ROOT / "images").mkdir(parents=True, exist_ok=True)
(STORAGE_ROOT / "documents").mkdir(parents=True, exist_ok=True)
(STORAGE_ROOT / "spreadsheets").mkdir(parents=True, exist_ok=True)


@router.post("/upload", response_model=List[FileObject])
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Handles file uploads, saves them to the appropriate storage directory,
    and returns their metadata.
    """
    file_objects = []
    for file in files:
        # Handle potential None for filename
        if not file.filename:
            continue  # Or raise an HTTPException for files without names

        # Handle potential None for content_type and detect file type by extension
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        # Determine file type based on extension and content type
        if file.content_type and file.content_type.startswith('image/'):
            file_type = 'image'
        elif file_extension in ['.csv', '.xlsx', '.xls']:
            file_type = 'spreadsheet'
        else:
            file_type = 'document'
        
        save_dir = STORAGE_ROOT / f"{file_type}s"
        file_path = save_dir / file.filename

        # Save the file asynchronously
        try:
            async with aio_open(file_path, 'wb') as out_file:
                while content := await file.read(1024):  # Read in chunks
                    await out_file.write(content)
        except Exception as e:
            # Handle potential file-saving errors
            raise HTTPException(status_code=500, detail=f"Could not save file: {e}")

        file_objects.append(FileObject(
            file_name=file.filename,
            file_path=str(file_path),  # Convert Path to string
            file_type=file_type
        ))
    return file_objects


@router.get("/files/{file_path:path}")
async def serve_file(file_path: str):
    """
    Serves uploaded files (images, documents) from the storage directory.
    """
    # Decode the file path
    file_path = unquote(file_path)
    
    # Security: ensure the path doesn't escape the storage directory
    if ".." in file_path or file_path.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid file path")
    
    # Resolve path relative to PROJECT_ROOT (so 'storage/...' work)
    full_path = PROJECT_ROOT / file_path
    
    # Check if file exists
    if not full_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
    # Determine media type based on file extension
    media_type, _ = guess_type(str(full_path))
    
    # Return the file
    return FileResponse(str(full_path), media_type=media_type)
