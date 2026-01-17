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
from typing import List
from pathlib import Path
import os

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
from .agent import DocumentAgent
# Import from backend schemas using module import to avoid conflict with local schemas.py
# Note: agents/document_agent/__init__.py -> agents/document_agent -> agents -> backend
import sys
from pathlib import Path as ImportPath
# Go up 2 levels: document_agent -> agents -> backend
backend_root = ImportPath(__file__).parent.parent.parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))
import schemas as backend_schemas
AgentResponse = backend_schemas.AgentResponse
AgentResponseStatus = backend_schemas.AgentResponseStatus
OrchestratorMessage = backend_schemas.OrchestratorMessage
DialogueContext = backend_schemas.DialogueContext
from typing import Optional, Dict

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
# DIALOGUE MANAGEMENT FOR ORCHESTRATOR INTEGRATION
# ============================================================================

# In-memory store for dialogue contexts
# Key: task_id, Value: DialogueContext
dialogue_store: Dict[str, DialogueContext] = {}

class DialogueManager:
    """Manages the state of bidirectional dialogues with orchestrator."""
    
    @staticmethod
    def get_context(task_id: str) -> Optional[DialogueContext]:
        """Get context for a paused task."""
        return dialogue_store.get(task_id)
    
    @staticmethod
    def create_context(task_id: str, agent_id: str = "document_agent") -> DialogueContext:
        """Create a new dialogue context."""
        context = DialogueContext(
            task_id=task_id,
            agent_id=agent_id,
            status="active"
        )
        dialogue_store[task_id] = context
        return context
    
    @staticmethod
    def pause_task(task_id: str, question: AgentResponse):
        """Pause a task and store the question."""
        if task_id in dialogue_store:
            dialogue_store[task_id].status = "paused"
            dialogue_store[task_id].current_question = question
    
    @staticmethod
    def resume_task(task_id: str):
        """Resume a paused task."""
        if task_id in dialogue_store:
            dialogue_store[task_id].status = "active"
    
    @staticmethod
    def complete_task(task_id: str):
        """Mark a task as completed."""
        if task_id in dialogue_store:
            dialogue_store[task_id].status = "completed"


# ============================================================================
# ORCHESTRATOR INTEGRATION ENDPOINT
# ============================================================================

@app.post("/execute", response_model=AgentResponse)
def execute_action(message: OrchestratorMessage):
    """
    Unified execution endpoint supporting bidirectional dialogue with orchestrator.
    
    Supports two modes:
    1. SPECIFIC ACTION: When 'action' is provided (e.g., '/edit'), executes that action directly
    2. COMPLEX PROMPT: When 'prompt' is provided, decomposes the request internally
    """
    try:
        agent = get_agent()
        payload = message.payload or {}
        
        # Generate or use existing task_id
        task_id = payload.get("task_id", f"task-{len(dialogue_store)}-doc")
        
        # Check for different message types
        action = message.action
        prompt = message.prompt
        answer = message.answer
        
        logger.info(f"[EXECUTE] Type={message.type} Action={action} Task={task_id}")
        
        # Initialize context
        context = DialogueManager.get_context(task_id)
        if not context:
            context = DialogueManager.create_context(task_id, "document_agent")
        
        # ==================== CONTINUE MODE (Resume from NEEDS_INPUT) ====================
        if message.type == "continue" and answer:
            logger.info(f"[CONTINUE] Resuming task {task_id} with user answer: {answer[:100]}")
            
            # Retrieve the pending question
            if context and context.current_question:
                pending_question = context.current_question
                
                # Get pending edit request from context
                pending_edit = context.current_question.context.get("pending_edit") if context.current_question.context else None
                
                if pending_edit:
                    # User approved or answered - proceed with edit using the answer
                    logger.info(f"[CONTINUE] Proceeding with approved edit, user answer: {answer}")
                    
                    # Create edit request from pending_edit, updating with user answer
                    edit_request = EditDocumentRequest(**pending_edit)
                    
                    # Call edit_document with user approval context
                    result = agent.edit_document(edit_request)
                    
                    DialogueManager.resume_task(task_id)
                    DialogueManager.complete_task(task_id)
                    
                    return AgentResponse(
                        status=result.get("status", AgentResponseStatus.COMPLETE),
                        result=result.get("result", "Edit completed with user approval"),
                        message=f"Successfully applied approved edits. User approval: {answer[:50]}...",
                        context={"task_id": task_id, "user_answer": answer}
                    )
            
            return AgentResponse(
                status=AgentResponseStatus.ERROR,
                result="No pending task found",
                message=f"Could not find paused task {task_id}",
                context={"task_id": task_id}
            )
        
        # ==================== CANCEL MODE ====================
        if message.type == "cancel":
            logger.info(f"[CANCEL] Cancelling task {task_id}")
            DialogueManager.complete_task(task_id)
            return AgentResponse(
                status=AgentResponseStatus.COMPLETE,
                result="Task cancelled",
                message=f"Task {task_id} has been cancelled",
                context={"task_id": task_id}
            )
        
        # ==================== CONTEXT_UPDATE MODE ====================
        if message.type == "context_update":
            logger.info(f"[CONTEXT_UPDATE] Updating context for task {task_id}")
            if context and message.additional_context:
                context.current_question.context = message.additional_context
            return AgentResponse(
                status=AgentResponseStatus.COMPLETE,
                result="Context updated",
                message=f"Context for task {task_id} has been updated",
                context={"task_id": task_id}
            )
        
        # ==================== EXECUTE MODE (New request) ====================
        if message.type == "execute" or not message.type:
            logger.info(f"[EXECUTE] Processing action {action}")
            
            # Route to appropriate action handler
            if action == "/analyze":
                request = AnalyzeDocumentRequest(**payload)
                result = agent.analyze_document(request)
                return AnalyzeDocumentResponse(**result)
            
            elif action == "/edit":
                request = EditDocumentRequest(**payload)
                result = agent.edit_document(request)
                
                # Check if result requires user approval
                if result.get("status") == AgentResponseStatus.NEEDS_INPUT:
                    # Store the pending edit request for continuation
                    DialogueManager.pause_task(task_id, AgentResponse(
                        status=AgentResponseStatus.NEEDS_INPUT,
                        question=result.get("question", "Approval required"),
                        question_type="confirmation",
                        context={
                            "task_id": task_id,
                            "pending_edit": payload,
                            "risk_assessment": result.get("risk_assessment")
                        }
                    ))
                    return AgentResponse(
                        status=AgentResponseStatus.NEEDS_INPUT,
                        question=result.get("question", "Approval required"),
                        question_type="confirmation",
                        context={
                            "task_id": task_id,
                            "pending_edit": payload,
                            "risk_assessment": result.get("risk_assessment")
                        }
                    )
                
                return EditDocumentResponse(**result)
            
            elif action == "/display":
                request = DisplayDocumentRequest(**payload)
                result = agent.display_document(request)
                return DisplayDocumentResponse(**result)
            
            elif action == "/create":
                request = CreateDocumentRequest(**payload)
                result = agent.create_document(request)
                return CreateDocumentResponse(**result)
            
            elif action == "/extract":
                request = ExtractDataRequest(**payload)
                result = agent.extract_data(request)
                return ExtractDataResponse(**result)
            
            else:
                return AgentResponse(
                    status=AgentResponseStatus.ERROR,
                    result="Unknown action",
                    message=f"Action '{action}' is not supported",
                    context={"task_id": task_id}
                )
        
        # Default response
        return AgentResponse(
            status=AgentResponseStatus.ERROR,
            result="Invalid message",
            message="Unable to process message",
            context={"task_id": task_id}
        )
    
    except Exception as e:
        logger.error(f"[EXECUTE] Error: {e}", exc_info=True)
        return AgentResponse(
            status=AgentResponseStatus.ERROR,
            result=str(e),
            message=f"Execution error: {str(e)}",
            context={"task_id": getattr(message.payload, "task_id", "unknown") if message.payload else "unknown"}
        )


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
