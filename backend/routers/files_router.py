"""
Files Router - Handles file upload and serving endpoints.

Extracted from main.py to improve code organization and maintainability.
"""

import io
import os
from typing import List, Optional
from fastapi import APIRouter, HTTPException, File, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from aiofiles import open as aio_open
from urllib.parse import unquote
from mimetypes import guess_type

from backend.schemas import FileObject

router = APIRouter(prefix="/api", tags=["Files"])

# Centralized storage paths
from backend.storage_config import STORAGE_ROOT, BACKEND_DIR

# Spreadsheet agent storage directory (resolved at import time)
_SPREADSHEET_STORAGE = STORAGE_ROOT / "spreadsheet_agent"


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
    
    # Resolve path relative to backend directory (so 'storage/...' works)
    full_path = BACKEND_DIR / file_path
    
    # Check if file exists
    if not full_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
    # Determine media type based on file extension
    media_type, _ = guess_type(str(full_path))
    
    # Return the file
    return FileResponse(str(full_path), media_type=media_type)


@router.get("/storage/{file_path:path}")
async def serve_storage_file(file_path: str):
    """
    Serves files from the shared storage root directory.
    Used by agents that save files under storage/ (documents, spreadsheets, etc.)
    """
    file_path = unquote(file_path)

    # Security: reject path traversal attempts
    if ".." in file_path or file_path.startswith("/") or file_path.startswith("\\"):
        raise HTTPException(status_code=400, detail="Invalid file path")

    full_path = (STORAGE_ROOT / file_path).resolve()

    # Confirm resolved path is still inside STORAGE_ROOT
    try:
        full_path.relative_to(STORAGE_ROOT.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid file path")

    if not full_path.exists() or not full_path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    media_type, _ = guess_type(str(full_path))
    filename = full_path.name
    return FileResponse(
        str(full_path),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/spreadsheet/download/{file_id}")
async def download_spreadsheet(
    file_id: str,
    format: Optional[str] = Query(default="xlsx", regex="^(xlsx|csv)$"),
):
    """
    Download a spreadsheet from the spreadsheet agent's storage.
    Supports on-the-fly format conversion between xlsx and csv.
    """
    file_id = unquote(file_id)

    # Security: reject path traversal
    if ".." in file_id or "/" in file_id or "\\" in file_id:
        raise HTTPException(status_code=400, detail="Invalid file id")

    # Search for the file - it may have been saved as xlsx or csv
    source_path = _SPREADSHEET_STORAGE / file_id
    if not source_path.exists():
        raise HTTPException(status_code=404, detail=f"Spreadsheet not found: {file_id}")

    source_ext = source_path.suffix.lower()
    target_format = (format or "xlsx").lower()

    # Fast path: requested format matches source format
    if (source_ext == ".xlsx" and target_format == "xlsx") or \
       (source_ext == ".csv" and target_format == "csv"):
        media_type = (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            if target_format == "xlsx" else "text/csv"
        )
        stem = source_path.stem
        return FileResponse(
            str(source_path),
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{stem}.{target_format}"'},
        )

    # Conversion path: use pandas to convert in memory
    try:
        import pandas as pd

        if source_ext == ".csv":
            df = pd.read_csv(source_path)
        else:
            df = pd.read_excel(source_path, engine="openpyxl")

        buf = io.BytesIO()
        stem = source_path.stem

        if target_format == "csv":
            csv_str = df.to_csv(index=False)
            return StreamingResponse(
                io.BytesIO(csv_str.encode("utf-8")),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f'attachment; filename="{stem}.csv"'
                },
            )
        else:  # xlsx
            df.to_excel(buf, index=False, engine="openpyxl")
            buf.seek(0)
            return StreamingResponse(
                buf,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={
                    "Content-Disposition": f'attachment; filename="{stem}.xlsx"'
                },
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Format conversion failed: {e}")
