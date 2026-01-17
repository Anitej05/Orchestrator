"""
Document Agent - FastAPI Server

Provides REST API endpoints for document operations.
Optimized for cloud deployment with minimal memory overhead.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import asyncio
from dotenv import load_dotenv
from aiofiles import open as aio_open
from fastapi import File, UploadFile
from typing import List, Dict, Optional
from pathlib import Path

# Import orchestrator-compatible schemas from backend root
import sys
BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
import schemas as backend_schemas

# Get workspace root
WORKSPACE_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
STORAGE_DIR = WORKSPACE_ROOT / "storage" / "documents"

from .schemas import (
    AnalyzeDocumentRequest, AnalyzeDocumentResponse,
    DisplayDocumentRequest, DisplayDocumentResponse,
    CreateDocumentRequest, CreateDocumentResponse,
    EditDocumentRequest, EditDocumentResponse,
    UndoRedoRequest, UndoRedoResponse,
    VersionHistoryRequest, VersionHistoryResponse,
    ExtractDataRequest, ExtractDataResponse
)
from . import agent as agent
from .agent import DocumentAgent

__all__ = ["DocumentAgent", "app"]
from .state import DialogueStateManager

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
# ORCHESTRATOR DIALOGUE STATE (PERSISTENT)
# ============================================================================

_dialogue_state = DialogueStateManager()


class DialogueManager:
    @staticmethod
    def get(task_id: str) -> Optional[backend_schemas.DialogueContext]:
        record = _dialogue_state.get(task_id)
        if not record:
            return None
        current_question = None
        if record.current_question:
            try:
                current_question = backend_schemas.AgentResponse(**record.current_question)
            except Exception:
                current_question = None
        return backend_schemas.DialogueContext(
            task_id=record.task_id,
            agent_id=record.agent_id,
            status=record.status,
            history=[],
            current_question=current_question,
        )

    @staticmethod
    def create(task_id: str, agent_id: str = "document_agent") -> backend_schemas.DialogueContext:
        record = _dialogue_state.get_or_create(task_id, agent_id)
        return backend_schemas.DialogueContext(
            task_id=record.task_id,
            agent_id=record.agent_id,
            status=record.status,
            history=[],
            current_question=None,
        )

    @staticmethod
    def pause(task_id: str, question: backend_schemas.AgentResponse) -> None:
        payload = question.model_dump() if hasattr(question, "model_dump") else question.dict()
        _dialogue_state.set_question(task_id, payload)

    @staticmethod
    def resume(task_id: str) -> None:
        _dialogue_state.update_status(task_id, "active")

    @staticmethod
    def complete(task_id: str) -> None:
        _dialogue_state.update_status(task_id, "completed")


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
# METRICS
# ============================================================================

@app.get("/metrics")
async def get_metrics():
    """Get current agent metrics including API calls, LLM calls, cache stats, and uptime."""
    try:
        agent = get_agent()
        return {
            "success": True,
            "metrics": agent.get_metrics()
        }
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/metrics/reset")
async def reset_metrics():
    """Reset all metrics counters to zero."""
    try:
        agent = get_agent()
        result = agent.reset_metrics()
        return {
            "success": True,
            **result
        }
    except Exception as e:
        logger.error(f"Failed to reset metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        
        uploaded_files = []
        for file in files:
            if not file.filename:
                continue
            
            # Save file to workspace root storage directory
            file_path = STORAGE_DIR / file.filename
            async with aio_open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            uploaded_files.append({
                "file_name": file.filename,
                "file_path": str(file_path),
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
        # Run sync operation in thread executor to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, agent.analyze_document, request)

        return AnalyzeDocumentResponse(
            success=bool(result.get('success')),
            answer=result.get('answer', ''),
            sources=result.get('sources'),
            status=result.get('status'),
            phase_trace=result.get('phase_trace'),
            grounding=result.get('grounding'),
            confidence=result.get('confidence'),
            review_required=result.get('review_required')
        )

    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
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
        # Run sync operation in thread executor to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, agent.create_document, request)

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
        # Run sync operation in thread executor to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, agent.edit_document, request)

        return EditDocumentResponse(
            success=bool(result.get('success')),
            message=result.get('message', ''),
            file_path=result.get('file_path', request.file_path),
            can_undo=result.get('can_undo', False),
            can_redo=result.get('can_redo', False),
            edit_summary=result.get('edit_summary'),
            status=result.get('status'),
            question=result.get('question'),
            question_type=result.get('question_type'),
            pending_plan=result.get('pending_plan'),
            risk_assessment=result.get('risk_assessment'),
            phase_trace=result.get('phase_trace')
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Edit error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ORCHESTRATOR INTEGRATION ENDPOINT
# ============================================================================

@app.post("/execute", response_model=backend_schemas.AgentResponse)
async def execute(message: backend_schemas.OrchestratorMessage):
    """Unified orchestrator endpoint supporting execute/continue/cancel/context_update."""
    agent = get_agent()
    payload = message.payload or {}
    task_id = payload.get("task_id") or f"doc-{int(asyncio.get_event_loop().time()*1000)}"

    ctx = DialogueManager.get(task_id) or DialogueManager.create(task_id)

    # CONTINUE
    if message.type == "continue":
        if not message.answer:
            return backend_schemas.AgentResponse(
                status=backend_schemas.AgentResponseStatus.ERROR,
                error="Missing answer for continue",
                context={"task_id": task_id},
            )
        if not ctx.current_question or not ctx.current_question.context:
            return backend_schemas.AgentResponse(
                status=backend_schemas.AgentResponseStatus.ERROR,
                error="No pending question for this task",
                context={"task_id": task_id},
            )

        pending_edit = ctx.current_question.context.get("pending_edit")
        if not pending_edit:
            return backend_schemas.AgentResponse(
                status=backend_schemas.AgentResponseStatus.ERROR,
                error="Missing pending edit payload",
                context={"task_id": task_id},
            )

        # Execute approved edit
        edit_req = EditDocumentRequest(**{**pending_edit, "auto_approve": True, "approval_response": message.answer})
        result = agent.edit_document(edit_req)
        DialogueManager.resume(task_id)
        DialogueManager.complete(task_id)
        return backend_schemas.AgentResponse(
            status=backend_schemas.AgentResponseStatus.COMPLETE if result.get("success") else backend_schemas.AgentResponseStatus.ERROR,
            result=result,
            error=None if result.get("success") else result.get("message"),
            context={"task_id": task_id},
        )

    # CANCEL
    if message.type == "cancel":
        DialogueManager.complete(task_id)
        return backend_schemas.AgentResponse(
            status=backend_schemas.AgentResponseStatus.COMPLETE,
            result={"cancelled": True},
            context={"task_id": task_id},
        )

    # CONTEXT UPDATE
    if message.type == "context_update":
        if message.additional_context:
            _dialogue_state.update_context(task_id, message.additional_context)
        return backend_schemas.AgentResponse(
            status=backend_schemas.AgentResponseStatus.COMPLETE,
            result={"context_updated": True},
            context={"task_id": task_id},
        )

    # EXECUTE
    action = message.action
    if action == "/edit":
        edit_req = EditDocumentRequest(**payload)
        result = agent.edit_document(edit_req)
        if result.get("status") == backend_schemas.AgentResponseStatus.NEEDS_INPUT.value:
            question = backend_schemas.AgentResponse(
                status=backend_schemas.AgentResponseStatus.NEEDS_INPUT,
                question=result.get("question"),
                question_type="confirmation",
                context={
                    "task_id": task_id,
                    "pending_edit": payload,
                    "pending_plan": result.get("pending_plan"),
                    "risk_assessment": result.get("risk_assessment"),
                },
            )
            DialogueManager.pause(task_id, question)
            return question

        return backend_schemas.AgentResponse(
            status=backend_schemas.AgentResponseStatus.COMPLETE if result.get("success") else backend_schemas.AgentResponseStatus.ERROR,
            result=result,
            error=None if result.get("success") else result.get("message"),
            context={"task_id": task_id},
        )

    if action == "/analyze":
        req = AnalyzeDocumentRequest(**payload)
        result = agent.analyze_document(req)
        return backend_schemas.AgentResponse(
            status=backend_schemas.AgentResponseStatus.COMPLETE if result.get("success") else backend_schemas.AgentResponseStatus.ERROR,
            result=result,
            error=None if result.get("success") else result.get("answer"),
            context={"task_id": task_id},
        )

    if action == "/extract":
        req = ExtractDataRequest(**payload)
        result = agent.extract_data(req)
        return backend_schemas.AgentResponse(
            status=backend_schemas.AgentResponseStatus.COMPLETE if result.get("success") else backend_schemas.AgentResponseStatus.ERROR,
            result=result,
            error=None if result.get("success") else result.get("message"),
            context={"task_id": task_id},
        )

    return backend_schemas.AgentResponse(
        status=backend_schemas.AgentResponseStatus.ERROR,
        error=f"Unsupported action: {action}",
        context={"task_id": task_id},
    )


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
