"""
Document Agent - FastAPI Server

Provides REST API endpoints for document operations.
Optimized for cloud deployment with minimal memory overhead.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
from dotenv import load_dotenv
from aiofiles import open as aio_open
from fastapi import File, UploadFile
from typing import List
from pathlib import Path

from .schemas import (
    AnalyzeDocumentRequest, AnalyzeDocumentResponse,
    DisplayDocumentRequest, DisplayDocumentResponse,
    CreateDocumentRequest, CreateDocumentResponse,
    EditDocumentRequest, EditDocumentResponse,
    UndoRedoRequest, UndoRedoResponse,
    VersionHistoryRequest, VersionHistoryResponse,
    ExtractDataRequest, ExtractDataResponse
)
from .agent import DocumentAgent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check required API keys
if not os.getenv('CEREBRAS_API_KEY'):
    logger.warning("CEREBRAS_API_KEY not set - some features may be limited")

# Initialize FastAPI app
app = FastAPI(
    title="Document Agent",
    description="Cloud-optimized document analysis and editing agent with RAG, LLM planning, and version control",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent (lazy loaded)
_agent = None

def get_agent() -> DocumentAgent:
    """Get or create agent instance."""
    global _agent
    if _agent is None:
        logger.info("Initializing DocumentAgent...")
        _agent = DocumentAgent()
    return _agent


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "document-agent",
        "version": "2.0.0"
    }


# ============================================================================
# FILE UPLOAD
# ============================================================================

@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload document files (PDF, DOCX, TXT) for processing.
    Returns file paths suitable for document operations.
    """
    try:
        os.makedirs("backend/storage/documents", exist_ok=True)
        
        uploaded_files = []
        for file in files:
            if not file.filename:
                continue
            
            # Save file
            file_path = os.path.join("backend/storage/documents", file.filename)
            async with aio_open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            uploaded_files.append({
                "file_name": file.filename,
                "file_path": file_path,
                "size_bytes": len(content),
                "ready": True
            })
            logger.info(f"✅ Uploaded: {file_path}")
        
        return {
            "success": True,
            "message": f"Uploaded {len(uploaded_files)} file(s)",
            "files": uploaded_files
        }
    
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# DOCUMENT ANALYSIS
# ============================================================================

@app.post("/analyze", response_model=AnalyzeDocumentResponse)
async def analyze_document(request: AnalyzeDocumentRequest):
    """
    Analyze documents using RAG and answer user queries.
    Requires FAISS vector store(s) to be created first.
    """
    try:
        agent = get_agent()
        result = agent.analyze_document(request)

        if not result.get('success'):
            raise HTTPException(
                status_code=400,
                detail=result.get('answer', 'Analysis failed')
            )

        return AnalyzeDocumentResponse(
            success=True,
            answer=result['answer'],
            sources=result.get('sources')
        )

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# DOCUMENT DISPLAY
# ============================================================================

@app.post("/display", response_model=DisplayDocumentResponse)
async def display_document(request: DisplayDocumentRequest):
    """
    Display a document with canvas rendering.
    Supports PDF, DOCX, and TXT files.
    """
    try:
        agent = get_agent()
        result = agent.display_document(request.file_path)

        if not result.get('success'):
            raise HTTPException(
                status_code=404,
                detail=result.get('message', 'Display failed')
            )

        return DisplayDocumentResponse(
            success=True,
            message=result['message'],
            canvas_display=result['canvas_display'],
            file_type=result.get('file_type', 'unknown')
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Display error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# DOCUMENT CREATION
# ============================================================================

@app.post("/create", response_model=CreateDocumentResponse)
async def create_document(request: CreateDocumentRequest):
    """
    Create a new document with specified content and format.
    Supports DOCX, PDF, and TXT formats.
    """
    try:
        agent = get_agent()
        result = agent.create_document(request)

        if not result.get('success'):
            raise HTTPException(
                status_code=400,
                detail=result.get('message', 'Creation failed')
            )

        return CreateDocumentResponse(
            success=True,
            message=result['message'],
            file_path=result['file_path']
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# DOCUMENT EDITING
# ============================================================================

@app.post("/edit", response_model=EditDocumentResponse)
async def edit_document(request: EditDocumentRequest):
    """
    Edit document using natural language instruction.
    Uses LLM to plan edits and applies them intelligently.
    """
    try:
        agent = get_agent()
        result = agent.edit_document(request)

        if not result.get('success'):
            raise HTTPException(
                status_code=400,
                detail=result.get('message', 'Edit failed')
            )

        return EditDocumentResponse(
            success=True,
            message=result['message'],
            file_path=result['file_path'],
            can_undo=result.get('can_undo', False),
            can_redo=result.get('can_redo', False),
            edit_summary=result.get('edit_summary')
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Edit error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# UNDO / REDO
# ============================================================================

@app.post("/undo-redo", response_model=UndoRedoResponse)
async def undo_redo(request: UndoRedoRequest):
    """
    Undo or redo document edits.
    Action must be 'undo' or 'redo'.
    """
    try:
        if request.action.lower() not in ['undo', 'redo']:
            raise HTTPException(
                status_code=400,
                detail="Action must be 'undo' or 'redo'"
            )

        agent = get_agent()
        result = agent.undo_redo(request)

        if not result.get('success'):
            raise HTTPException(
                status_code=400,
                detail=result.get('message', f'{request.action.capitalize()} failed')
            )

        return UndoRedoResponse(
            success=True,
            message=result['message'],
            file_path=result['file_path'],
            can_undo=result.get('can_undo', False),
            can_redo=result.get('can_redo', False)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Undo/redo error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# VERSION HISTORY
# ============================================================================

@app.post("/versions", response_model=VersionHistoryResponse)
async def get_versions(request: VersionHistoryRequest):
    """
    Get version history for a document.
    Returns all saved versions with metadata.
    """
    try:
        agent = get_agent()
        result = agent.get_version_history(request.file_path)

        if not result.get('success'):
            raise HTTPException(
                status_code=400,
                detail=result.get('message', 'Failed to get history')
            )

        return VersionHistoryResponse(
            success=True,
            message=result['message'],
            versions=result.get('versions', []),
            current_version=result.get('current_version', -1)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Version history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# DATA EXTRACTION
# ============================================================================

@app.post("/extract", response_model=ExtractDataResponse)
async def extract_data(request: ExtractDataRequest):
    """
    Extract structured data from document.
    Types: 'text' (summary), 'tables' (table data), 'structured' (key information)
    """
    try:
        if request.extraction_type not in ['text', 'tables', 'structured']:
            raise HTTPException(
                status_code=400,
                detail="extraction_type must be 'text', 'tables', or 'structured'"
            )

        agent = get_agent()
        result = agent.extract_data(request)

        if not result.get('success'):
            raise HTTPException(
                status_code=400,
                detail=result.get('message', 'Extraction failed')
            )

        return ExtractDataResponse(
            success=True,
            message=result['message'],
            extracted_data=result.get('extracted_data', {}),
            data_format=result.get('data_format', request.extraction_type)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# STARTUP / SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize agent on startup."""
    logger.info("Document Agent starting...")
    get_agent()
    logger.info("Document Agent ready")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Document Agent shutting down...")
    # Cleanup can be added here if needed
