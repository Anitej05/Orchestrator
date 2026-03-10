# Orbimesh Backend/main.py
import uuid
import logging
import json
import asyncio
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from fastapi import UploadFile, File
from aiofiles import open as aio_open


import os
import subprocess
import sys
import platform
import socket
import re

# Ensure both import styles resolve without requiring external PYTHONPATH.
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
for _path in (BACKEND_DIR, PROJECT_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(BACKEND_DIR, ".env"), override=True)

# --- Third-party Imports ---
from fastapi import FastAPI, HTTPException, Depends, Query, WebSocket, WebSocketDisconnect, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import httpx
# Lazy import SentenceTransformer to avoid jaxlib dependency issues
# from sentence_transformers import SentenceTransformer

# --- Local Application Imports ---
CONVERSATION_HISTORY_DIR = "conversation_history"
from database import SessionLocal, get_db
from models import Agent, StatusEnum, Workflow, WorkflowExecution, UserThread, WorkflowSchedule, WorkflowWebhook
from backend.schemas import ProcessRequest, ProcessResponse, PlanResponse, FileObject, ActionApprovalRequest, ActionRejectRequest
from backend.orchestrator.graph import create_graph_with_checkpointer, create_execution_subgraph, serialize_complex_object
from langgraph.checkpoint.memory import MemorySaver
from routers import connect_router
from utils.mega_logger import setup_mega_logger, mega_log_block
from routers import credentials_router
from routers import integrations_router
from routers import webhooks_router

# --- Lifespan Event Handler (replaces deprecated @app.on_event) ---
# Global task reference for health checker
_health_checker_task: Optional[asyncio.Task] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Async context manager for FastAPI lifespan events.
    Startup logic runs before yield, shutdown logic runs after.
    """
    global _health_checker_task
    
    # === STARTUP ===
    # Run database migrations automatically
    try:
        logger.info("🔧 Running database migrations...")
        import subprocess
        result = subprocess.run(
            ["alembic", "upgrade", "head"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            logger.info("✅ Database migrations applied")
        else:
            logger.warning(f"⚠️ Migration warnings: {result.stderr}")
    except Exception as e:
        logger.error(f"❌ Failed to run migrations: {str(e)}", exc_info=True)
    
    # Create database tables if they don't exist
    try:
        logger.info("🔧 Ensuring database tables exist...")
        from services.agent_registry_service import create_tables
        create_tables()
        logger.info("✅ Database tables ready")
    except Exception as e:
        logger.error(f"❌ Failed to create tables: {str(e)}", exc_info=True)
    
    # Sync agent definitions from SKILL.md files (UAP) to database
    try:
        from services.agent_registry_service import sync_skill_entries
        logger.info("Syncing SKILL.md agent definitions to database...")
        result = sync_skill_entries(verbose=True)
        if result.get('errors'):
            logger.warning(f"SKILL.md sync completed with {len(result['errors'])} error(s)")
        else:
            logger.info("✅ SKILL.md agent sync completed successfully")
    except Exception as e:
        logger.error(f"Failed to sync SKILL.md entries: {str(e)}", exc_info=True)
    
    # Clear agent logs from previous runs for clean startup
    try:
        logs_dir = Path("logs")
        if logs_dir.exists():
            for agent_log in logs_dir.glob("*_agent.log"):
                try:
                    agent_log.unlink()  # Delete old agent logs
                    logger.debug(f"Cleared old agent log: {agent_log.name}")
                except Exception as e:
                    logger.debug(f"Could not clear {agent_log.name}: {e}")
            logger.info("✓ Agent logs cleared for fresh startup")
    except Exception as e:
        logger.debug(f"Error clearing agent logs: {e}")
    
    # Start agents in background
    start_agents_async()
    logger.info("✓ Agents started in background")
    
    # Start background health checker
    _health_checker_task = asyncio.create_task(check_agent_health_background())
    logger.info("✓ Health checker started")
    
    # Initialize workflow scheduler and load active schedules
    try:
        logger.info("Initializing workflow scheduler...")
        from backend.services.workflow_scheduler import init_scheduler
        from database import SessionLocal
        db = SessionLocal()
        try:
            scheduler = init_scheduler(db)
            jobs = scheduler.scheduler.get_jobs()
            logger.info(f"✓ Workflow scheduler initialized with {len(jobs)} jobs loaded")
            for job in jobs:
                logger.info(f"  - Job: {job.id} | Next run: {job.next_run_time}")
        finally:
            db.close()
    except Exception as e:
        logger.error(f"✗ Failed to initialize workflow scheduler: {str(e)}", exc_info=True)
    
    logger.info("=" * 60)
    logger.info("APPLICATION STARTUP COMPLETED")
    logger.info("=" * 60)
    
    yield  # App is running
    
    # === SHUTDOWN ===
    logger.info("=" * 60)
    logger.info("APPLICATION SHUTTING DOWN")
    logger.info("=" * 60)
    
    # Cancel background health checker
    if _health_checker_task:
        logger.info("✓ Cancelling health checker task...")
        _health_checker_task.cancel()
        try:
            await _health_checker_task
        except asyncio.CancelledError:
            pass
        logger.info("✓ Health checker stopped")
    
    # Stop all agent processes
    logger.info("✓ Stopping agent processes...")
    cleanup_agents()
    logger.info("✓ Agent processes stopped")
    
    # Shutdown workflow scheduler
    try:
        from backend.services.workflow_scheduler import get_scheduler
        scheduler = get_scheduler()
        logger.info("✓ Shutting down workflow scheduler...")
        scheduler.shutdown()
        logger.info("✓ Workflow scheduler stopped")
    except Exception as e:
        logger.error(f"✗ Error shutting down scheduler: {str(e)}")
    
    logger.info("=" * 60)
    logger.info("APPLICATION SHUTDOWN COMPLETE")
    logger.info("=" * 60)

# --- App Initialization and Configuration ---
app = FastAPI(
    title="Unified Agent Service API",
    version="1.0",
    description="An API for both finding/managing agents and orchestrating tasks.",
    lifespan=lifespan
)

# ================================================================================================= #
# 🚀 Configure Global Mega Logger Strategy (Writes every orchestrator and API event to daily logs)
# ================================================================================================= #

logger = setup_mega_logger("OrchestratorAPI")

def clear_orchestrator_log():
    """Clear the orchestrator log file for a new conversation.
    WARNING: Now deprecated because MegaLogger persists everything forever across days. 
    We just log a distinct boundary instead.
    """
    mega_log_block(logger, "NEW CONVERSATION STARTED", "Starting a fresh thread in the Mega Logger.", level=logging.INFO)

# Initialize memory for persistent conversations
checkpointer = MemorySaver()

# Create the main graph and execution subgraph with the checkpointer
graph = create_graph_with_checkpointer(checkpointer)
execution_subgraph = create_execution_subgraph(checkpointer)

# Simple in-memory conversation store as backup
conversation_store: Dict[str, Dict[str, Any]] = {}
from threading import Lock
store_lock = Lock()

# Canvas live updates: Maps thread_id -> canvas data for browser agent visualization
live_canvas_updates: Dict[str, Dict[str, Any]] = {}
canvas_lock = Lock()

# WebSocket screenshot relay: Maps thread_id -> frontend WebSocket for direct screenshot streaming
# Browser agent sends screenshots via /ws/screenshots/{thread_id}, orchestrator relays to frontend
frontend_websockets: Dict[str, WebSocket] = {}
frontend_ws_lock = asyncio.Lock()  # Async lock for WebSocket registry

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],  # Expose all headers
)

# Include routers
app.include_router(connect_router.router)
app.include_router(credentials_router.router)
app.include_router(integrations_router.router)
app.include_router(webhooks_router.router)  # Composio OAuth webhook events

# Import and include content management router
from routers import content_router
app.include_router(content_router.router)

# Import and include the new modular routers
from routers import agents_router, conversations_router, workflows_router

# Inject shared state into routers that need it
# conversations_router needs access to conversation_store and store_lock
conversations_router.set_shared_state(conversation_store, store_lock)
# workflows_router needs access to checkpointer
workflows_router.set_checkpointer(checkpointer)

# NOTE: files_router, workflows_router, and dashboard_router are NOT included here
# because their routes are already defined inline in main.py (with additional logic).
# Including them would cause Duplicate Operation ID warnings.
# TODO: Migrate inline routes to routers and re-enable these includes.
# app.include_router(files_router.router)
# app.include_router(workflows_router.router)
# app.include_router(dashboard_router.router)

app.include_router(agents_router.router)
app.include_router(conversations_router.router)


# --- Static Files for Screenshots ---
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Create storage directory if it doesn't exist
# Fix: Use absolute path to project root (Orbimesh/storage) instead of relative (Orbimesh/backend/storage)
storage_path = (Path(__file__).parent.parent / "storage").resolve()
storage_path.mkdir(exist_ok=True)
# Images moved to strict hierarchy (storage/system/images or browser_agent/screenshots)

# Mount storage directory for serving screenshots
app.mount("/storage", StaticFiles(directory=str(storage_path)), name="storage")
logger.info(f"Mounted /storage for serving screenshot files from {storage_path}")


# ============================================================================
# SECURE FILE SERVING ROUTE
# ============================================================================
# Auth System: Uses Clerk JWT authentication via auth.py module
# - Validates JWT tokens using JWKS (JSON Web Key Sets) from Clerk
# - Requires "Authorization: Bearer <token>" header on all requests
# - Function: get_user_from_request(request) verifies token and returns user info
# - Returns 401 if token is missing, invalid, or expired
#
# This route serves agent-generated files from backend/storage with:
# - Path traversal protection (rejects ../, absolute paths)
# - Smart filename lookup across storage folders
# - Auto MIME type detection with inline preview support
# - Newest version selection when file exists in multiple folders
# ============================================================================

@app.get("/files/{file_path:path}", tags=["Files"])
async def serve_file(file_path: str, request: Request):
    """
    Securely serve user files (documents, images, spreadsheets) with inline preview support.
    
    This endpoint allows the orchestrator UI to display or download agent-generated content.
    Files are served with Content-Disposition: inline when supported by browsers (PDFs, images, text).
    
    **Authentication:** Requires Clerk JWT token via Authorization header
    
    **Path Handling:**
    - Relative path: /files/documents/report.pdf → backend/storage/documents/report.pdf
    - Filename only: /files/report.pdf → searches in: documents, content, images, spreadsheets, etc.
    - Multi-folder: Returns newest version by modification timestamp if found in multiple locations
    
    **Security:**
    - Path traversal blocked (../, ~, //)
    - All resolved paths validated to stay within backend/storage
    - Symlink escape prevention via resolve() validation
    
    **Supported File Types:**
    - Documents: .pdf, .docx, .txt
    - Spreadsheets: .xlsx, .csv
    - Images: .png, .jpg, .gif, .svg
    - Data: .json
    
    Args:
        file_path: Relative path from storage root or just filename
        request: FastAPI request object for auth validation
        
    Returns:
        FileResponse with appropriate Content-Type and Content-Disposition headers
        
    Raises:
        401: Missing or invalid authentication token
        404: File not found in storage
        500: Server error (logged but details not exposed)
        
    Examples:
        GET /files/documents/sample.pdf
        GET /files/sample.pdf  (auto-detects folder)
        GET /files/images/chart.png
        GET /files/spreadsheets/data.xlsx
    """
    try:
        # ===== AUTHENTICATION =====
        # Validate Clerk JWT token using existing auth pattern
        from auth import get_user_from_request
        
        try:
            user = get_user_from_request(request)
            user_id = user.get("sub") or user.get("user_id") or user.get("id")
            logger.info(f"File request authenticated: user_id={user_id}, file={file_path}")
        except HTTPException as auth_error:
            logger.warning(f"File auth failed for {file_path}: {auth_error.detail}")
            raise HTTPException(
                status_code=401,
                detail="Authentication required. Please provide a valid Bearer token."
            )
        
        # ===== SECURITY VALIDATION =====
        from backend.utils.file_server import is_safe_path, find_file_in_storage, get_mime_type, should_inline_preview
        
        # Check for path traversal attempts
        if not is_safe_path(file_path):
            logger.warning(f"Unsafe path rejected: {file_path}")
            raise HTTPException(
                status_code=400,
                detail="Invalid file path. Path traversal attempts are not allowed."
            )
        
        # ===== FILE LOOKUP =====
        # Smart search across storage folders with newest version selection
        result = find_file_in_storage(file_path)
        if not result:
            logger.info(f"File not found: {file_path}")
            raise HTTPException(
                status_code=404,
                detail=f"File not found: {file_path}"
            )
        
        resolved_path, relative_path = result
        logger.info(f"Serving file: {relative_path} -> {resolved_path}")
        
        # ===== MIME TYPE & DISPOSITION =====
        mime_type = get_mime_type(resolved_path)
        
        # Use inline preview for browser-compatible types, otherwise download
        if should_inline_preview(mime_type):
            disposition = "inline"
        else:
            disposition = f'attachment; filename="{resolved_path.name}"'
        
        # ===== SERVE FILE =====
        return FileResponse(
            path=str(resolved_path),
            media_type=mime_type,
            headers={
                "Content-Disposition": disposition,
                "X-File-Path": relative_path,  # Debug header showing resolved path
            }
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions (auth, not found, validation)
        raise
    
    except Exception as e:
        # Log unexpected errors but don't expose details to client
        logger.error(f"File serving error for {file_path}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error while serving file"
        )


# Example requests (for testing):
# GET http://localhost:8000/files/documents/sample.pdf
# GET http://localhost:8000/files/sample.pdf   -> auto-detect folder
# GET http://localhost:8000/files/images/chart.png
# GET http://localhost:8000/files/spreadsheets/data.xlsx


# ============================================================================
# DOCUMENT CREATION ROUTE (UNIFIED)
# ============================================================================
# This route wraps the document agent's /create endpoint and provides:
# - Clerk authentication (same pattern as other routes)
# - Request validation and error handling
# - Canvas preview generation for newly created documents
# - Integration with the file serving route (/files)
# ============================================================================

from pydantic import Field

class CreateDocumentRequest(BaseModel):
    """Request to create a new document via orchestrator."""
    content: str = Field(..., description="Document content")
    file_name: str = Field(..., description="Filename (e.g., 'report.docx')")
    file_type: str = Field(default="docx", description="Type: docx, pdf, or txt")
    thread_id: Optional[str] = Field(None, description="Conversation thread ID")

class CreateDocumentResponse(BaseModel):
    """Response with created document details."""
    success: bool
    message: str
    file_path: str
    relative_path: Optional[str] = None  # Path for /files endpoint
    canvas_display: Optional[Dict[str, Any]] = None
    preview_url: Optional[str] = None  # URL to preview via /files

@app.post("/api/documents/create", tags=["Document Operations"])
async def create_document_unified(request: CreateDocumentRequest, req: Request):
    """
    Create a new document through the orchestrator.
    
    This is a unified endpoint that:
    1. Validates Clerk authentication
    2. Calls document agent's /create endpoint
    3. Generates inline preview for newly created file
    4. Returns file path suitable for /files endpoint
    
    Args:
        request: CreateDocumentRequest with content, filename, and type
        req: FastAPI request for authentication
        
    Returns:
        CreateDocumentResponse with file_path and canvas_display for immediate preview
        
    Example:
        POST /api/documents/create
        {
            "content": "# Report\n\nThis is a test report.",
            "file_name": "report.docx",
            "file_type": "docx",
            "thread_id": "conv-123"
        }
        
        Response:
        {
            "success": true,
            "message": "Created report.docx",
            "file_path": "/backend/storage/documents/report.docx",
            "relative_path": "documents/report.docx",
            "preview_url": "/files/documents/report.docx",
            "canvas_display": { ... }
        }
    """
    try:
        # ===== AUTHENTICATION =====
        # Allow internal orchestrator calls (X-Internal-Request header)
        is_internal = req.headers.get("X-Internal-Request") == "true"
        
        from auth import get_user_from_request
        user_id = "system"  # default for internal calls
        
        if not is_internal:
            try:
                user = get_user_from_request(req)
                user_id = user.get("sub") or user.get("user_id") or user.get("id")
                logger.info(f"Document creation requested by: user_id={user_id}")
            except HTTPException:
                raise HTTPException(
                    status_code=401,
                    detail="Authentication required"
                )
        else:
            logger.info("Internal orchestrator request - bypassing auth")
        
        # ===== CALL DOCUMENT AGENT =====
        import httpx
        
        # Call document agent's /create endpoint
        agent_url = os.getenv("DOCUMENT_AGENT_URL", "http://localhost:8070")
        async with httpx.AsyncClient(timeout=30.0) as client:
            agent_request = {
                "content": request.content,
                "file_name": request.file_name,
                "file_type": request.file_type,
                "output_dir": "storage/document_agent",
                "thread_id": request.thread_id
            }
            
            try:
                agent_response = await client.post(
                    f"{agent_url}/create",
                    json=agent_request
                )
                agent_response.raise_for_status()
            except httpx.RequestError as e:
                logger.error(f"Document agent error: {e}")
                raise HTTPException(
                    status_code=503,
                    detail="Document agent unavailable"
                )
        
        agent_data = agent_response.json()
        if not agent_data.get("success"):
            raise HTTPException(
                status_code=400,
                detail=agent_data.get("message", "Document creation failed")
            )
        
        file_path = agent_data.get("file_path", "")
        
        # ===== GENERATE PREVIEW =====
        from pathlib import Path as PathlibPath
        
        canvas_display = None
        relative_path = None
        preview_url = None
        
        if file_path and PathlibPath(file_path).exists():
            # Get relative path for /files endpoint
            try:
                abs_path = PathlibPath(file_path).resolve()
                storage_base = (PathlibPath(__file__).parent.parent / "storage").resolve()
                relative_path = str(abs_path.relative_to(storage_base))
                preview_url = f"/files/{relative_path}"
                
                # Generate canvas display for inline preview
                file_ext = PathlibPath(file_path).suffix.lower()
                
                if file_ext in ['.pdf']:
                    # PDF preview
                    canvas_display = {
                        "canvas_type": "pdf",
                        "file_path": file_path,
                        "file_name": PathlibPath(file_path).name,
                        "preview_url": preview_url
                    }
                elif file_ext in ['.docx', '.doc']:
                    # DOCX preview (convert to PDF for display)
                    canvas_display = {
                        "canvas_type": "docx",
                        "file_path": file_path,
                        "file_name": PathlibPath(file_path).name,
                        "preview_url": preview_url,
                        "note": "Click preview URL to view document"
                    }
                elif file_ext in ['.txt']:
                    # Text preview
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content_preview = f.read()[:2000]
                    canvas_display = {
                        "canvas_type": "text",
                        "file_name": PathlibPath(file_path).name,
                        "content": content_preview,
                        "full_preview_url": preview_url
                    }
                else:
                    canvas_display = {
                        "canvas_type": "file",
                        "file_name": PathlibPath(file_path).name,
                        "preview_url": preview_url
                    }
            except Exception as e:
                logger.warning(f"Could not generate canvas preview: {e}")
        
        return CreateDocumentResponse(
            success=True,
            message=agent_data.get("message", f"Created {request.file_name}"),
            file_path=file_path,
            relative_path=relative_path,
            canvas_display=canvas_display,
            preview_url=preview_url
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document creation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to create document"
        )


# ============================================================================
# SPREADSHEET CREATION ROUTE (UNIFIED)
# ============================================================================
# Similar to document creation but for spreadsheets
# ============================================================================

class CreateSpreadsheetRequest(BaseModel):
    """Request to create a new spreadsheet via orchestrator."""
    filename: str = Field(..., description="Spreadsheet filename (e.g., 'data.xlsx')")
    data: Dict[str, Any] = Field(..., description="Data structure: {columns: [...], rows: [...]}")
    file_format: str = Field(default="xlsx", description="Format: xlsx or csv")
    thread_id: Optional[str] = Field(None, description="Conversation thread ID")

class CreateSpreadsheetResponse(BaseModel):
    """Response with created spreadsheet details."""
    success: bool
    message: str
    file_path: str
    relative_path: Optional[str] = None
    canvas_display: Optional[Dict[str, Any]] = None
    preview_url: Optional[str] = None

@app.post("/api/spreadsheets/create", tags=["Spreadsheet Operations"])
async def create_spreadsheet_unified(request: CreateSpreadsheetRequest, req: Request):
    """
    Create a new spreadsheet through the orchestrator.
    
    Args:
        request: CreateSpreadsheetRequest with filename, data, and format
        req: FastAPI request for authentication
        
    Returns:
        CreateSpreadsheetResponse with file_path and canvas_display
        
    Example:
        POST /api/spreadsheets/create
        {
            "filename": "sales_report.xlsx",
            "file_format": "xlsx",
            "data": {
                "columns": ["Month", "Sales", "Profit"],
                "rows": [
                    ["Jan", 10000, 2000],
                    ["Feb", 12000, 2500]
                ]
            },
            "thread_id": "conv-123"
        }
    """
    try:
        # ===== AUTHENTICATION =====
        # Allow internal orchestrator calls (X-Internal-Request header)
        is_internal = req.headers.get("X-Internal-Request") == "true"
        
        from auth import get_user_from_request
        user_id = "system"  # default for internal calls
        
        if not is_internal:
            try:
                user = get_user_from_request(req)
                user_id = user.get("sub") or user.get("user_id") or user.get("id")
                logger.info(f"Spreadsheet creation requested by: user_id={user_id}")
            except HTTPException:
                raise HTTPException(status_code=401, detail="Authentication required")
        else:
            logger.info("Internal orchestrator request - bypassing auth")
        
        # ===== CREATE SPREADSHEET =====
        import pandas as pd
        from pathlib import Path as PathlibPath
        
        # Convert data format to DataFrame
        try:
            if "rows" in request.data and "columns" in request.data:
                df = pd.DataFrame(request.data["rows"], columns=request.data["columns"])
            else:
                df = pd.DataFrame(request.data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid data format: {str(e)}")
        
        # Create storage directory
        storage_dir = PathlibPath(__file__).parent.parent / "storage" / "spreadsheet_agent"
        storage_dir.mkdir(parents=True, exist_ok=True)
        file_path = storage_dir / request.filename
        
        # Save file
        try:
            if request.file_format.lower() == "csv":
                df.to_csv(file_path, index=False)
            else:  # Default to xlsx
                df.to_excel(file_path, index=False, engine='openpyxl')
            logger.info(f"Created spreadsheet: {file_path}")
        except Exception as e:
            logger.error(f"File creation error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create file: {str(e)}")
        
        # ===== GENERATE PREVIEW =====
        storage_root = (PathlibPath(__file__).parent.parent / "storage").resolve()
        try:
             relative_path = str(file_path.resolve().relative_to(storage_root))
        except ValueError:
             # Fallback if path resolution fails (should not happen with correct setup)
             relative_path = str(file_path.name)
             logger.warning(f"Could not resolve relative path for {file_path}")
        preview_url = f"/files/{relative_path}"
        
        # Canvas display with data preview
        canvas_display = {
            "canvas_type": "spreadsheet",
            "file_name": request.filename,
            "columns": df.columns.tolist(),
            "rows": df.head(10).values.tolist(),  # First 10 rows
            "total_rows": len(df),
            "preview_url": preview_url
        }
        
        return CreateSpreadsheetResponse(
            success=True,
            message=f"Created {request.filename}",
            file_path=str(file_path),
            relative_path=relative_path,
            canvas_display=canvas_display,
            preview_url=preview_url
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Spreadsheet creation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create spreadsheet")

# --- Sentence Transformer Model Loading ---
# Lazy load to avoid jaxlib dependency issues at startup
model = None

def get_sentence_transformer_model():
    """Lazy load the sentence transformer model only when needed"""
    global model
    if model is None:
        # Disable JAX to avoid jaxlib dependency issues
        os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
        os.environ['JAX_PLATFORMS'] = ''  # Disable JAX backend
        
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-mpnet-base-v2')
    return model

# --- Interactive Conversation Models ---
class UserResponse(BaseModel):
    """Model for user responses to orchestrator questions"""
    response: str
    thread_id: str
    files: Optional[List[FileObject]] = None

class ConversationStatus(BaseModel):
    """Model for conversation status responses"""
    thread_id: str
    status: str  # "completed", "pending_user_input", "error"
    question_for_user: Optional[str] = None

class ScheduleWorkflowRequest(BaseModel):
    """Model for scheduling workflow requests"""
    cron_expression: str
    input_template: Dict[str, Any] = {}

class UpdateScheduleRequest(BaseModel):
    """Model for updating schedule requests"""
    is_active: Optional[bool] = None
    cron_expression: Optional[str] = None
    input_template: Optional[Dict[str, Any]] = None
    final_response: Optional[str] = None
    task_agent_pairs: Optional[List[Dict]] = None
    error_message: Optional[str] = None

# --- Agent Server Startup ---
def is_port_in_use(port: int) -> bool:
    """Checks if a local port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def wait_for_port(port: int, agent_file: str, timeout: int = 15):
    """Waits for a network port to become active."""
    start_time = time.time()
    logger.info(f"Waiting for agent '{agent_file}' to start on port {port}...")
    while time.time() - start_time < timeout:
        if is_port_in_use(port):
            logger.info(f"Agent '{agent_file}' is now running on port {port}.")
            return True
        time.sleep(0.5)
    logger.error(f"Agent '{agent_file}' did not start on port {port} within {timeout} seconds.")
    return False

def start_agent_servers():
    """
    Finds and starts agent servers, with enhanced logging and better error handling
    to track which agents start successfully and which fail.
    """
    global agent_processes
    
    # Use absolute path based on the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))  # This gets the backend directory
    agents_dir = os.path.join(project_root, "agents")
    
    if not os.path.isdir(agents_dir):
        logger.warning(f"'{agents_dir}' directory not found. Skipping agent server startup.")
        return

    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    logger.info(f"Agent logs will be stored in the '{logs_dir}' directory.")

    started_agents = []
    failed_agents = []

    # UAP: Read from SKILL.md definitions (Source of Truth)
    agent_configs = {}
    
    # SKILL.md directories to scan (each contains SKILL.md with port)
    skill_dirs = [
        ("spreadsheet_agent", "spreadsheet_agent/__init__.py"),
        ("mail_agent", "mail_agent/agent.py"),
        ("gmail_agent", "gmail_agent/__init__.py"),
        ("browser_agent", "browser_agent/__init__.py"),
        ("document_agent_lib", "document_agent_lib/__init__.py"),
        ("zoho_books", "zoho_books/zoho_books_agent.py"),
    ]
    
    import yaml
    for skill_dir, script_path in skill_dirs:
        skill_file = os.path.join(agents_dir, skill_dir, "SKILL.md")
        full_script_path = os.path.join(agents_dir, script_path)
        
        if not os.path.exists(skill_file):
            continue
            
        if not os.path.exists(full_script_path):
            logger.warning(f"Script not found for {skill_dir}: {script_path}")
            continue
            
        try:
            with open(skill_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Extract YAML frontmatter
                if content.startswith('---'):
                    parts = content.split('---', 2)
                    if len(parts) >= 3:
                        yaml_content = parts[1]
                        config = yaml.safe_load(yaml_content)
                        port = config.get('port')
                        if port:
                            agent_configs[script_path] = int(port)  # Store relative path, not full path
                            logger.info(f"UAP: Found {skill_dir} on port {port}")
        except Exception as e:
            logger.warning(f"Failed to parse SKILL.md for {skill_dir}: {e}")

    logger.info(f"Target Agent Configurations: {len(agent_configs)} agents")

    for agent_file, port in agent_configs.items():
        agent_path = os.path.join(agents_dir, agent_file)
        # Extract agent name from path (e.g., "spreadsheet_agent/__init__.py" -> "spreadsheet_agent")
        agent_name = agent_file.split('/')[0].split('\\')[0]
        
        if is_port_in_use(port):
            logger.info(f"Agent '{agent_name}' is already running on port {port}.")
            started_agents.append({
                'agent': agent_name,
                'port': port,
                'status': 'already_running'
            })
            continue

        logger.info(f"Attempting to start '{agent_name}' on port {port}...")
        log_path = os.path.join(logs_dir, f"{agent_name}.log")

        # Create the subprocess
        process = None
        try:
            with open(log_path, 'w') as log_file:
                if platform.system() == "Windows":
                    process = subprocess.Popen(
                        [sys.executable, agent_path],
                        stdout=log_file,
                        stderr=log_file,
                        creationflags=subprocess.CREATE_NO_WINDOW
                    )
                else:
                    process = subprocess.Popen(
                        [sys.executable, agent_path],
                        stdout=log_file,
                        stderr=log_file
                    )
                agent_processes.append(process)
        except Exception as e:
            logger.error(f"Failed to start {agent_name}: {e}")
            failed_agents.append({
                'agent': agent_name,
                'reason': f'Failed to spawn process: {e}',
                'port': port
            })
            continue

        # Wait for the port to be in use with a timeout
        # With lazy imports and reload=False, agents should start quickly
        agent_timeout = 15  # Reasonable timeout for fast startup
        
        if wait_for_port(port, agent_name, timeout=agent_timeout):
            logger.info(f"Successfully started agent '{agent_name}' on port {port}")
            started_agents.append({
                'agent': agent_name,
                'port': port,
                'status': 'started',
                'process': process
            })
        else:
            logger.error(f"Timed out waiting for agent '{agent_name}' to start on port {port}")
            failed_agents.append({
                'agent': agent_name,
                'reason': f'Timed out waiting for port {port} to become available',
                'port': port
            })
            if process:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()



    # Log summary of agent startup results
    logger.info(f"Agent startup completed. Started: {len(started_agents)}, Failed: {len(failed_agents)}")
    
    if started_agents:
        logger.info("Successfully started agents:")
        for agent_info in started_agents:
            logger.info(f"  - {agent_info['agent']} on port {agent_info['port']} ({agent_info['status']})")
    
    if failed_agents:
        logger.error("Failed to start agents:")
        for agent_info in failed_agents:
            logger.error(f"  - {agent_info['agent']}: {agent_info['reason']}")
        
        # Provide detailed instructions for manual startup
        logger.info("To start agents manually, run these commands in separate terminals:")
        for agent_info in failed_agents:
            agent_file = agent_info['agent']
            agent_path = os.path.join(agents_dir, agent_file)
            logger.info(f"  - python {agent_path}")
    
    logger.info("Agent startup check completed.")

os.makedirs("storage/images", exist_ok=True)
os.makedirs("storage/documents", exist_ok=True)

@app.post("/api/upload", response_model=List[FileObject])
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Handles file uploads, saves them to the appropriate storage directory,
    and returns their metadata.
    """
    file_objects = []
    for file in files:
        # **FIX 1: Handle potential None for filename**
        if not file.filename:
            logger.warning("File upload skipped: filename is None or empty")
            continue

        try:
            # **FIX 2: Handle potential None for content_type and detect file type by extension**
            file_extension = os.path.splitext(file.filename)[1].lower()
            
            # Determine file type based on extension and content type
            if file.content_type and file.content_type.startswith('image/'):
                file_type = 'image'
            elif file_extension in ['.csv', '.xlsx', '.xls']:
                file_type = 'spreadsheet'
            else:
                file_type = 'document'
            
            # **FIX 3: Use pathlib for cross-platform path handling**
            from pathlib import Path as PathlibPath
            save_dir = PathlibPath(__file__).parent / "storage" / f"{file_type}s"
            
            # Create storage directory if it doesn't exist
            try:
                save_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"Failed to create storage directory {save_dir}: {e}")
                raise HTTPException(status_code=500, detail=f"Could not create storage directory: {e}")
            
            # **FIX 4: Create full file path and normalize to forward slashes**
            file_path_obj = save_dir / file.filename
            # Convert to string and normalize to forward slashes for cross-platform compatibility
            file_path = str(file_path_obj).replace('\\', '/')
            
            logger.info(f"Uploading file: {file.filename} -> {file_path} (type: {file_type})")

            # Save the file asynchronously (use the normalized path)
            try:
                async with aio_open(str(file_path_obj), 'wb') as out_file:  # Use original Path for actual file I/O
                    while content := await file.read(1024*64):  # Read in larger chunks (64KB)
                        await out_file.write(content)
            except Exception as e:
                # Handle potential file-saving errors
                logger.error(f"Failed to save file {file.filename} to {file_path}: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Could not save file: {str(e)}")
            
            # CRITICAL VALIDATION: Verify file was actually saved before returning
            if not file_path_obj.exists():
                raise HTTPException(
                    status_code=500, 
                    detail=f"File save failed: {file.filename} does not exist at {file_path}"
                )
            
            # Check file size
            file_size = file_path_obj.stat().st_size
            logger.info(f"File uploaded successfully: {file.filename} ({file_size} bytes) at {file_path}")

            file_objects.append(FileObject(
                file_name=file.filename,
                file_path=file_path,  # Return normalized path with forward slashes
                file_type=file_type
            ))
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error uploading {file.filename}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Unexpected error uploading file: {str(e)}")
    
    logger.info(f"Upload complete: {len(file_objects)} file(s) uploaded successfully")
    return file_objects

@app.get("/api/files/{file_path:path}")
async def serve_file(file_path: str):
    """
    Serves uploaded files (images, documents) from the storage directory.
    """
    # Decode the file path
    from urllib.parse import unquote
    file_path = unquote(file_path)
    
    # Security: ensure the path doesn't escape the storage directory
    if ".." in file_path or file_path.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid file path")
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # Determine media type based on file extension
    from mimetypes import guess_type
    media_type, _ = guess_type(file_path)
    
    # Return the file
    from fastapi.responses import FileResponse
    return FileResponse(file_path, media_type=media_type)

# --- Unified Orchestration Service ---
async def execute_orchestration(
    prompt: Optional[str],
    thread_id: str,
    user_response: Optional[str] = None,
    files: Optional[List[FileObject]] = None,
    stream_callback=None,
    task_event_callback=None,
    planning_mode: bool = False,
    owner: Optional[Any] = None
):
    """
    Unified orchestration logic that correctly persists and merges file context
    across all turns in a conversation. Simplified and more robust version.
    
    Args:
        owner: Dict with user ownership info, e.g. {"user_id": "...", "email": "..."}.
               Only provided on initial conversation start or when explicitly passed.
    """
    normalized_owner = owner
    if isinstance(owner, str):
        normalized_owner = {"user_id": owner}
    elif owner and not isinstance(owner, dict):
        normalized_owner = None

    owner = normalized_owner
    logger.info(f"Starting orchestration for thread_id: {thread_id}, planning_mode: {planning_mode}, owner={owner}")
    
    # PHASE 5: Debug uploaded files state at entry
    if files:
        logger.info(f"📂 EXECUTE_ORCHESTRATION: Received {len(files)} files")
        for idx, f in enumerate(files):
            logger.info(f"  File {idx+1}: {f.file_name} (type={f.file_type}, path={f.file_path})")
    else:
        logger.info("📂 EXECUTE_ORCHESTRATION: No files received in this request")

    # Build config with task_event_callback and owner if provided
    # Keep graph recursion budget above Brain.max_iterations so controller-level
    # safeguards can terminate gracefully before LangGraph hard-recursion errors.
    config = {"recursion_limit": 80, "configurable": {"thread_id": thread_id}}
    if task_event_callback:
        config["configurable"]["task_event_callback"] = task_event_callback
        logger.info("✅ Task event callback registered for real-time streaming")
    if owner:
        config["configurable"]["owner"] = owner
        logger.info(f"✅ Owner information provided for orchestration: {owner}")

    # Get the current state of the conversation from the in-memory store first (most recent)
    # Fall back to checkpointer if not in memory
    with store_lock:
        current_conversation = conversation_store.get(thread_id)
    
    if not current_conversation:
        # If not in memory, try checkpointer
        current_checkpoint = checkpointer.get(config)
        # Extract the state from the checkpoint if it exists
        # The checkpoint structure is { "values": State, "next": List[str], "config": RunnableConfig }
        current_conversation = current_checkpoint.get("values", {}) if current_checkpoint else {}
    
    # If still no conversation data, try loading from JSON file (for saved workflows)
    if not current_conversation:
        history_path = os.path.join(CONVERSATION_HISTORY_DIR, f"{thread_id}.json")
        if os.path.exists(history_path):
            try:
                with open(history_path, "r", encoding="utf-8") as f:
                    json_data = json.load(f)
                    
                    # Convert message dicts to LangChain message objects
                    messages = []
                    for msg_data in json_data.get("messages", []):
                        try:
                            msg_type = msg_data.get("type", "").lower()
                            content = msg_data.get("content", "")
                            metadata = msg_data.get("metadata", {})
                            msg_id = msg_data.get("id")
                            timestamp = msg_data.get("timestamp")
                            
                            # Prepare additional_kwargs
                            additional_kwargs = metadata.copy() if metadata else {}
                            if timestamp:
                                additional_kwargs['timestamp'] = timestamp
                            if msg_id:
                                additional_kwargs['id'] = msg_id
                            
                            if msg_type in ["user", "human"]:
                                messages.append(HumanMessage(content=content, additional_kwargs=additional_kwargs))
                            elif msg_type in ["assistant", "ai"]:
                                messages.append(AIMessage(content=content, additional_kwargs=additional_kwargs))
                            else:
                                messages.append(SystemMessage(content=content, additional_kwargs=additional_kwargs))
                        except Exception as msg_err:
                            logger.warning(f"Failed to convert message: {msg_err}")
                    
                    # Extract the saved workflow data
                    current_conversation = {
                        "thread_id": json_data.get("thread_id"),
                        "original_prompt": json_data.get("original_prompt", "") or json_data.get("metadata", {}).get("original_prompt", ""),
                        "task_plan": json_data.get("task_plan", []) or json_data.get("plan", []),
                        "task_agent_pairs": json_data.get("task_agent_pairs", []),
                        "needs_approval": json_data.get("needs_approval", False),
                        "plan_approved": json_data.get("plan_approved", False),
                        "approval_required": json_data.get("approval_required", False),
                        "pending_user_input": json_data.get("pending_user_input", False),
                        "question_for_user": json_data.get("question_for_user"),
                        "status": json_data.get("status"),
                        "messages": messages,
                        "completed_tasks": json_data.get("completed_tasks", []),
                        "uploaded_files": json_data.get("uploaded_files", []),
                        "parsed_tasks": json_data.get("parsed_tasks", []) or json_data.get("metadata", {}).get("parsed_tasks", [])
                    }
                    logger.info(f"Loaded workflow data from JSON: needs_approval={current_conversation.get('needs_approval')}, task_plan_count={len(current_conversation.get('task_plan', []))}, messages_count={len(messages)}")
            except Exception as e:
                logger.error(f"Failed to load conversation JSON for {thread_id}: {e}")
                current_conversation = {}

    # --- State Initialization ---
    if user_response:
        # Continuing an interactive workflow where the user answered a question
        logger.info(f"Resuming conversation for thread_id: {thread_id} with user response.")
        logger.info(f"USER RESPONSE BRANCH: user_response='{user_response}', planning_mode={planning_mode}")
        initial_state = dict(current_conversation)  # Convert to dict if it's a State object
        initial_state["user_response"] = user_response
        # initial_state["pending_user_input"] = False # REMOVED: Must preserve this flag from history for resume logic to work
        initial_state["question_for_user"] = None
        initial_state["parse_retry_count"] = 0
        
        # Extract user_id from owner if available
        if owner and isinstance(owner, dict) and owner.get("user_id"):
            initial_state["user_id"] = owner["user_id"]
            initial_state["owner_id"] = owner["user_id"]  # Add both for compatibility
        
        # Handle plan approval responses
        needs_approval = initial_state.get("needs_approval", False)
        task_plan = initial_state.get("task_plan", [])
        print(f"!!! USER RESPONSE: needs_approval={needs_approval}, response='{user_response}', task_plan_count={len(task_plan)} !!!")
        logger.info(f"Checking approval state: needs_approval={needs_approval}, user_response='{user_response}', task_plan_count={len(task_plan)}")
        
        if needs_approval:
            user_response_lower = user_response.lower().strip()
            logger.info(f"Processing approval response: '{user_response_lower}'")
            
            if user_response_lower in ["approve", "yes", "proceed", "continue", "execute", "go", "ok"]:
                print("!!! USER APPROVED - Setting planning_mode=False and plan_approved=True !!!")
                logger.info(f"User APPROVED execution plan for thread_id: {thread_id}")
                logger.info("Clearing approval flags and continuing execution")
                # Simply turn off planning mode and clear approval flags
                initial_state["needs_approval"] = False
                initial_state["approval_required"] = False
                initial_state["planning_mode"] = False  # This is the key - let it run normally now
                initial_state["pending_user_input"] = False
                initial_state["question_for_user"] = None
                initial_state["plan_approved"] = True  # NEW: Flag to skip validation and go straight to execution
                logger.info(f"State after approval: needs_approval={initial_state['needs_approval']}, planning_mode={initial_state['planning_mode']}, pending_user_input={initial_state['pending_user_input']}, plan_approved={initial_state.get('plan_approved')}")
                # Don't modify original_prompt - keep everything as is
            elif user_response_lower in ["cancel", "no", "stop", "abort", "reject"]:
                logger.info(f"User CANCELLED execution plan for thread_id: {thread_id}")
                print("!!! USER CANCELLED - Stopping execution !!!")
                initial_state["final_response"] = "Execution cancelled by user."
                return initial_state
            else:
                # User wants to modify the plan - treat as a new prompt with modifications
                logger.info(f"User wants to MODIFY plan with: '{user_response}'")
                print("!!! USER MODIFICATION REQUEST - Replanning with modifications !!!")
                print(f"!!! CURRENT STATE HAS original_prompt: {'original_prompt' in initial_state}")
                if "original_prompt" in initial_state:
                    print(f"!!! ORIGINAL PROMPT VALUE: '{initial_state['original_prompt'][:200]}...'")
                
                # Add user's modification message to the conversation history
                from backend.orchestrator.message_manager import MessageManager
                existing_messages = initial_state.get("messages", [])
                modification_message = HumanMessage(content=user_response)
                updated_messages = MessageManager.add_message(
                    existing_messages, 
                    modification_message
                )
                initial_state["messages"] = updated_messages
                
                # Combine original prompt with the modification in a way that makes the intent clear
                if "original_prompt" in initial_state and initial_state["original_prompt"]:
                    # Make it clear this is a modification/addition to the existing workflow
                    initial_state["original_prompt"] = f"{initial_state['original_prompt']}\n\n{user_response}"
                    print(f"!!! COMBINED PROMPT: '{initial_state['original_prompt'][:300]}...'")
                else:
                    initial_state["original_prompt"] = user_response
                    print("!!! NO ORIGINAL PROMPT - USING USER RESPONSE AS NEW PROMPT !!!")
                
                # Clear the plan and restart planning with the modified prompt
                initial_state["task_plan"] = []
                initial_state["task_agent_pairs"] = []
                initial_state["needs_approval"] = False
                initial_state["approval_required"] = False
                initial_state["pending_user_input"] = False
                initial_state["question_for_user"] = None
                initial_state["planning_mode"] = True  # Keep in planning mode to regenerate plan
                initial_state["plan_approved"] = False
                # Also clear completed tasks to start fresh
                initial_state["completed_tasks"] = []
                initial_state["latest_completed_tasks"] = []
                logger.info(f"Restarting planning with modified prompt: '{initial_state['original_prompt']}'")
        else:
            # Regular user response - add to context
            if "original_prompt" in initial_state:
                initial_state["original_prompt"] = f"{initial_state['original_prompt']}\n\nAdditional context: {user_response}"
            else:
                initial_state["original_prompt"] = user_response
            
        # Clear any previous final response to avoid confusion
        initial_state["final_response"] = None

    elif prompt and current_conversation:
        # A new prompt is sent in an existing conversation thread
        logger.info("NEW PROMPT IN EXISTING CONVERSATION BRANCH")
        logger.info(f"Continuing conversation for thread_id: {thread_id} with new prompt, planning_mode: {planning_mode}")
        logger.info(f"Prompt content: '{prompt[:100]}'")
        
        # Use MessageManager to add new message without duplicates
        from backend.orchestrator.message_manager import MessageManager
        existing_messages = current_conversation.get("messages", [])
        # Create message with metadata to preserve timestamp and ID
        import hashlib
        timestamp = time.time()
        unique_string = f"human:{prompt}:{timestamp}"
        msg_id = hashlib.md5(unique_string.encode()).hexdigest()[:16]
        new_user_message = HumanMessage(
            content=prompt,
            additional_kwargs={"timestamp": timestamp, "id": msg_id}
        )
        updated_messages = MessageManager.add_message(existing_messages, new_user_message)
        logger.info(f"Continuing conversation. Total messages: {len(updated_messages)}")
        
        # Check if this is a plan approval - if so, preserve the plan
        needs_approval = current_conversation.get("needs_approval", False)
        is_plan_approval = needs_approval and user_response and user_response.lower().strip() in ["approve", "yes", "proceed", "continue", "execute", "go", "ok"]
        
        initial_state = {
            # Carry over essential long-term memory from the previous turn
            "messages": updated_messages,
            "completed_tasks": current_conversation.get("completed_tasks", []),  # Preserve completed tasks for context
            "uploaded_files": current_conversation.get("uploaded_files", []), # Persist files

            # Reset short-term memory for the new task (UNLESS it's a plan approval)
            "original_prompt": prompt,
            "parsed_tasks": current_conversation.get("parsed_tasks", []) if is_plan_approval else [],
            "user_expectations": current_conversation.get("user_expectations", {}) if is_plan_approval else {},
            "candidate_agents": current_conversation.get("candidate_agents", {}) if is_plan_approval else {},
            "task_agent_pairs": current_conversation.get("task_agent_pairs", []) if is_plan_approval else [],
            "task_plan": current_conversation.get("task_plan", []) if is_plan_approval else [],
            "final_response": None,
            "pending_user_input": False,
            "question_for_user": None,
            "user_response": None,
            "parsing_error_feedback": None,
            "parse_retry_count": 0,
            "needs_complex_processing": None,  # Let analyze_request determine this
            "analysis_reasoning": None,
            "planning_mode": planning_mode,  # Set planning mode from parameter
            "plan_approved": current_conversation.get("plan_approved", False) if is_plan_approval else False,

            # Preserve canvas confirmation fields (needed to resume execution on confirm)
            "pending_confirmation": current_conversation.get("pending_confirmation", False),
            "pending_confirmation_task": current_conversation.get("pending_confirmation_task"),
            "canvas_requires_confirmation": current_conversation.get("canvas_requires_confirmation", False),
            "canvas_confirmation_message": current_conversation.get("canvas_confirmation_message"),

            # Preserve Canvas Registry state across turns
            "canvas_registry": current_conversation.get("canvas_registry"),
            "active_canvas_id": current_conversation.get("active_canvas_id"),
            # Backward compat canvas fields
            "has_canvas": current_conversation.get("has_canvas", False),
            "canvas_type": current_conversation.get("canvas_type"),
            "canvas_content": current_conversation.get("canvas_content"),
            "canvas_data": current_conversation.get("canvas_data"),
            "canvas_title": current_conversation.get("canvas_title"),

            # --- NEW ORCHESTRATOR FIELDS ---
            # Reset todo_list/execution state fresh for each new turn so the planner
            # creates a new task plan rather than carrying over the previous turn's completed tasks.
            "todo_list": [],
            "memory": current_conversation.get("memory", {}),
            "iteration_count": 0,
            "failure_count": 0,
            "max_iterations": 25,
            "action_history": current_conversation.get("action_history", []),
            "insights": current_conversation.get("insights", {}),
            "execution_plan": None,
            "current_phase_id": None,
            "pending_approval": current_conversation.get("pending_approval", False),
            "pending_decision": current_conversation.get("pending_decision"),
            # If the user approved an action via the REST endpoint, carry the approved
            # decision + flag forward so the brain skips re-planning and hands executes
            # it directly. Without this, brain re-plans from scratch and picks a
            # different agent, causing an infinite approval loop.
            "pending_action_approval": current_conversation.get("pending_action_approval", False),
            "decision": current_conversation.get("decision") if current_conversation.get("pending_action_approval") else None,
        }
        
        # Extract user_id from owner if available
        if owner and isinstance(owner, dict) and owner.get("user_id"):
            initial_state["user_id"] = owner["user_id"]
            initial_state["owner_id"] = owner["user_id"]  # Add both for compatibility
    
    elif current_conversation:
        # Resuming without a new prompt or user response (e.g., status check)
        initial_state = dict(current_conversation) # Convert to dict if it's a State object
        logger.info(f"Checking status for thread_id: {thread_id}")

    else:
        # A brand new conversation
        if not prompt:
            raise ValueError("Prompt is required for new conversations")
        logger.info("NEW CONVERSATION BRANCH")
        logger.info(f"Starting new conversation for thread_id: {thread_id}, planning_mode: {planning_mode}")
        
        # Clear orchestrator log for new conversation
        clear_orchestrator_log()
        
        # Create message with metadata to preserve timestamp and ID
        import hashlib
        timestamp = time.time()
        unique_string = f"human:{prompt}:{timestamp}"
        msg_id = hashlib.md5(unique_string.encode()).hexdigest()[:16]
        
        initial_state = {
            "original_prompt": prompt,
            "messages": [HumanMessage(
                content=prompt,
                additional_kwargs={"timestamp": timestamp, "id": msg_id}
            )],
            "uploaded_files": [], # Start with an empty file list
            "parsed_tasks": [],
            "user_expectations": {},
            "candidate_agents": {},
            "task_agent_pairs": [],
            "task_plan": [],
            "completed_tasks": [],
            "final_response": None,
            "pending_user_input": False,
            "question_for_user": None,
            "user_response": None,
            "parsing_error_feedback": None,
            "parse_retry_count": 0,
            "needs_complex_processing": None,  # Let analyze_request determine this
            "analysis_reasoning": None,
            "planning_mode": planning_mode,  # Set planning mode from parameter
            
            # --- NEW ORCHESTRATOR FIELDS ---
            "todo_list": [],
            "memory": {},
            "iteration_count": 0,
            "failure_count": 0,
            "max_iterations": 25,
            "action_history": [],
            "insights": {},
            "execution_plan": None,
            "current_phase_id": None,
            "pending_approval": False,
            "pending_decision": None,
            "pending_action_approval": False,
        }
        
        # Extract user_id from owner if available
        if owner and isinstance(owner, dict) and owner.get("user_id"):
            initial_state["user_id"] = owner["user_id"]
            initial_state["owner_id"] = owner["user_id"]  # Add both for compatibility
        

        # --- NEW THREAD DB PERSISTENCE ---
        if owner and owner.get("user_id"):
            try:
                from database import get_db_session
                from models import UserThread
                with get_db_session() as db:
                    # Check if thread already exists
                    existing = db.query(UserThread).filter_by(thread_id=thread_id).first()
                    if not existing:
                        logger.info(f"Creating new UserThread record in SQL for {thread_id} (user: {owner['user_id']})")
                        # Create title from prompt (max 100 chars)
                        clean_title = (prompt[:97] + "...") if len(prompt) > 100 else prompt
                        new_thread = UserThread(
                            thread_id=thread_id,
                            user_id=owner["user_id"],
                            title=clean_title
                        )
                        db.add(new_thread)
                        db.commit()
            except Exception as e:
                logger.error(f"Failed to save new UserThread for {thread_id}: {e}")

    # --- File Merging Logic ---
    # This block runs for EVERY turn, ensuring new files are always added to the state
    # CRITICAL: Preserve file_id from previous turns to maintain spreadsheet agent state
    if files:
        # Use a dictionary keyed by file_path to merge lists and avoid duplicates
        file_map = {f['file_path']: f for f in initial_state.get("uploaded_files", [])}
        
        # IMPORTANT: Reset is_current_turn for all existing files
        # This ensures only newly uploaded files are marked as current
        for file_path in file_map:
            file_map[file_path]['is_current_turn'] = False
        
        for new_file in files:
            new_file_dict = new_file.model_dump()
            file_path = new_file.file_path
            
            # Mark this file as uploaded in the current turn
            new_file_dict['is_current_turn'] = True
            logger.info(f"📎 Marking file as current turn: {new_file.file_name}")
            
            # CRITICAL FIX: If we already have this file with a file_id, preserve it
            # This ensures modifications made by the spreadsheet agent are not lost
            if file_path in file_map:
                existing_file = file_map[file_path]
                existing_file_id = existing_file.get('file_id') or existing_file.get('content_id')
                new_file_id = new_file_dict.get('file_id') or new_file_dict.get('content_id')
                
                # If new file doesn't have a file_id but existing one does, preserve it
                if existing_file_id and not new_file_id:
                    logger.info(f"📝 Preserving existing file_id '{existing_file_id}' for file: {file_path}")
                    new_file_dict['file_id'] = existing_file_id
                    # Also preserve content_id if that was the identifier
                    if existing_file.get('content_id') and not new_file_dict.get('content_id'):
                        new_file_dict['content_id'] = existing_file.get('content_id')
            
            file_map[file_path] = new_file_dict
        
        initial_state["uploaded_files"] = list(file_map.values())
        logger.info(f"File context updated. Total unique files in state: {len(initial_state['uploaded_files'])}")
    
    # Store the prepared initial state before running the graph
    with store_lock:
        conversation_store[thread_id] = initial_state.copy()

    final_state = None
    try:
        # Determine if we should use execution subgraph or main graph
        is_post_approval = user_response is not None and initial_state.get("plan_approved") == True
        
        # Select the appropriate graph
        if is_post_approval:
            logger.info("Graph execution mode: POST-APPROVAL EXECUTION using subgraph")
            print("!!! GRAPH EXECUTION: POST-APPROVAL - Using execution subgraph !!!")
            selected_graph = execution_subgraph
            graph_input = initial_state
        else:
            logger.info("Graph execution mode: NORMAL execution using main graph")
            print("!!! GRAPH EXECUTION: NORMAL - Using main graph !!!")
            selected_graph = graph
            graph_input = initial_state
        
        if stream_callback:
            # Streaming mode for WebSocket
            node_count = 0
            # Track all nodes dynamically instead of using a fixed list
            total_nodes_estimate = 5 if is_post_approval else 11
            
            async for event in selected_graph.astream(graph_input, config=config, stream_mode="updates"):
                for node_name, node_output in event.items():
                    node_count += 1
                    # Calculate progress based on actual node count with a reasonable estimate
                    # Use 90% as max during execution, reserve 10% for completion
                    progress = min((node_count / total_nodes_estimate) * 90, 90)
                    if isinstance(node_output, dict):
                        final_state = {**final_state, **node_output} if final_state else node_output
                    await stream_callback(node_name, node_output, progress, node_count, thread_id)
                    if isinstance(node_output, dict) and node_output.get("pending_user_input"):
                        logger.info(f"Workflow paused for user input in thread_id: {thread_id}")
                        break
            
            # Send final 100% progress after all nodes complete
            if final_state and not (isinstance(final_state, dict) and final_state.get("pending_user_input")):
                await stream_callback("workflow_complete", {"status": "completed"}, 100, node_count, thread_id)
            
            # After streaming, get the actual state if not available
            # After streaming, we MUST fetch the full authoritative state from the graph
            # The 'final_state' accumulated above only contains deltas (updates from the last node)
            # which means 'messages' would be incomplete (only new messages).
            try:
                state_snapshot = await selected_graph.aget_state(config)
                final_state = state_snapshot.values
                logger.info(f"✅ Retrieved full authoritative state from graph. Messages count: {len(final_state.get('messages', []))}")
            except Exception as e:
                logger.warning(f"Failed to get full state snapshot, falling back to accumulated state: {e}")
                if not final_state:
                    final_state = await selected_graph.ainvoke(graph_input, config=config)
        else:
            # Single response mode for HTTP
            final_state = await selected_graph.ainvoke(graph_input, config=config)

        # Ensure thread_id is always present in state for persistence
        if isinstance(final_state, dict):
            final_state["thread_id"] = thread_id
        else:
            logger.warning(f"final_state for {thread_id} is not a dict ({type(final_state)}); persistence may be limited")

        # Store the final state after the graph run
        with store_lock:
            conversation_store[thread_id] = final_state.copy()
            
        # Save conversation history using orchestrator's save routine with owner info
        try:
            from backend.orchestrator.graph import save_conversation_history
            
            # Extract owner_id from owner parameter
            owner_id = None
            if owner:
                if isinstance(owner, str):
                    owner_id = owner
                else:
                    owner_id = owner.get("user_id") or owner.get("sub") or owner.get("id")
            
            # Add owner to final_state and config for save
            if owner_id:
                owner_obj = {"user_id": owner_id}
                final_state["owner"] = owner_obj
                save_conversation_history(final_state, {"configurable": {"thread_id": thread_id, "owner": owner_obj}})
                logger.info(f"Conversation history saved in execute_orchestration for thread_id {thread_id} with owner_id {owner_id}")
                
                # ─── Register / refresh UserThread in DB so conversation appears in sidebar ───
                try:
                    from database import SessionLocal as _SessionLocal
                    from models import UserThread
                    _db = _SessionLocal()
                    try:
                        existing_thread = _db.query(UserThread).filter_by(thread_id=thread_id).first()
                        if existing_thread:
                            # Update updated_at so conversation moves to top of list
                            existing_thread.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
                            logger.info(f"Updated UserThread timestamp for {thread_id} in execute_orchestration")
                        else:
                            # Generate a title from the original prompt
                            raw_title = (
                                final_state.get("original_prompt")
                                or (prompt if prompt else "")
                                or "New Conversation"
                            )
                            conversation_title = raw_title[:100].strip()
                            new_thread = UserThread(
                                thread_id=thread_id,
                                user_id=owner_id,
                                title=conversation_title,
                            )
                            _db.add(new_thread)
                            logger.info(f"Created UserThread for {thread_id} in execute_orchestration (user={owner_id}, title='{conversation_title}')")
                        _db.commit()
                    finally:
                        _db.close()
                except Exception as db_err:
                    logger.error(f"Failed to register UserThread for {thread_id} in execute_orchestration: {db_err}")
                    # Non-fatal — conversation is already saved to file
            else:
                # Save without owner (will still save but without user association)
                save_conversation_history(final_state, {"configurable": {"thread_id": thread_id}})
                logger.warning(f"Conversation history saved in execute_orchestration for thread_id {thread_id} WITHOUT owner_id")
        except Exception as e:
            logger.error(f"Failed to save conversation history for {thread_id}: {e}")

        # Plan file saving is handled by orchestrator checkpointer


        logger.info(f"Orchestration completed for thread_id: {thread_id}")

        # === ARTIFACT DISTILLATION: Distill learnings from this conversation ===
        try:
            from backend.orchestrator.content_orchestrator import hooks as content_hooks
            user_id = final_state.get("user_id", "default")
            await content_hooks.on_conversation_end(final_state, thread_id, user_id)
        except Exception as e:
            logger.debug(f"Artifact distillation skipped: {e}")

        return final_state

    except Exception as e:
        error_msg = str(e)
        if "No valid tasks to process" in error_msg or "No tasks to rank" in error_msg or "Halting: No agents found for task ''" in error_msg:
            logger.warning(f"No valid tasks could be parsed from prompt for thread_id {thread_id}. Original prompt: '{prompt}'")
            error_state = {
                "final_response": f"I couldn't identify any specific tasks from your message: '{prompt}'. Could you please be more specific?",
                "pending_user_input": False,
                "question_for_user": None,
            }
            with store_lock:
                conversation_store[thread_id] = {**initial_state, **error_state}
            return conversation_store[thread_id]

        logger.error(f"Error during orchestration for thread_id {thread_id}: {e}", exc_info=True)
        raise

def process_node_data(node_name: str, node_output, progress: float, node_count: int, thread_id: str = None):
    """Extract meaningful data from node output consistently"""
    serializable_data = {}
    
    # Map node names to user-friendly stage names and descriptions
    stage_mapping = {
        "omni_brain": {"stage": "planning", "message": "Brain is analyzing state..."},
        "omni_hands": {"stage": "executing", "message": "Hands executing action..."},
        "action_approval_required": {"stage": "approval_required", "message": "Waiting for action approval..."},
        "manage_todo_list": {"stage": "planning", "message": "Brain is analyzing tasks..."},
        "execute_next_action": {"stage": "executing", "message": "Executing task..."},
        "analyze_request": {"stage": "analyzing", "message": "Analyzing your request..."},
        "parse_prompt": {"stage": "parsing", "message": "Breaking down your request into tasks..."},
        "agent_directory_search": {"stage": "searching", "message": "Searching for capable agents (REST & MCP)..."},
        "rank_agents": {"stage": "ranking", "message": "Ranking and selecting best agents..."},
        "plan_execution": {"stage": "planning", "message": "Creating execution plan..."},
        "validate_plan_for_execution": {"stage": "validating", "message": "Validating execution plan..."},
        "execute_batch": {"stage": "executing", "message": "Executing tasks with agents..."},
        "evaluate_agent_response": {"stage": "evaluating", "message": "Evaluating agent responses..."},
        "generate_final_response": {"stage": "aggregating", "message": "Generating final response..."},
        "save_history": {"stage": "finalizing", "message": "Saving conversation history..."},
        "load_history": {"stage": "loading", "message": "Loading conversation history..."},
    }
    
    stage_info = stage_mapping.get(node_name, {"stage": "processing", "message": f"Processing {node_name.replace('_', ' ')}..."})
    
    node_specific_data = {
        "progress_percentage": round(progress, 1),
        "node_sequence": node_count,
        "current_stage": stage_info["stage"],
        "stage_message": stage_info["message"]
    }

    if isinstance(node_output, dict):
        for key, value in node_output.items():
            logger.debug(f"Processing key '{key}' from node '{node_name}'")
            serializable_data[key] = serialize_complex_object(value)

            # Extract node-specific meaningful data
            
            # --- NEW MEGA LOG RECORD ---
            mega_log_block(
                logger, 
                f"NODE EXECUTION | {node_name.upper()} | Thread: {thread_id}", 
                f"Progress: {progress}%\nStage: {stage_info['stage']}\nKey modified: {key}\nPayload Preview:\n{str(value)[:1000]}",
                level=logging.INFO
            )
            
            # --- NEW ORCHESTRATOR NODES ---
            if node_name in ["omni_brain", "manage_todo_list"] and key == "todo_list":
                 if value:
                    # 'value' is a list of TaskItem dicts
                    task_names = []
                    tasks_completed = 0
                    tasks_total = len(value)
                    
                    for task in value:
                        if isinstance(task, dict):
                            t_desc = task.get("description", "Unknown Task")
                            t_status = task.get("status")
                            task_names.append(t_desc)
                            if t_status == "completed":
                                tasks_completed += 1
                        
                    node_specific_data["tasks_identified"] = tasks_total
                    node_specific_data["tasks_completed_count"] = tasks_completed
                    node_specific_data["task_names"] = task_names
                    node_specific_data["description"] = f"Brain managing {tasks_total} tasks ({tasks_completed} done)"
            
            # Extract adaptive planning fields for real-time UI updates
            elif node_name == "omni_brain" and key in ["execution_plan", "action_history", "insights"]:
                if key == "action_history" and isinstance(value, list):
                    # Deduplicate action_history to prevent frontend duplicate display
                    seen_actions = set()
                    unique_history = []
                    for action in value:
                        if isinstance(action, dict):
                            # Create unique identifier from key fields
                            action_id = f"{action.get('iteration')}-{action.get('resource_id')}-{action.get('action_type')}-{action.get('execution_time_ms')}"
                            if action_id not in seen_actions:
                                seen_actions.add(action_id)
                                # Strip large result fields before sending to frontend
                                cleaned_action = {k: v for k, v in action.items() if k not in ['result_raw', 'result_full']}
                                unique_history.append(cleaned_action)
                            else:
                                logger.debug(f"⚠️ Filtered duplicate action: {action_id}")
                        else:
                            unique_history.append(action)
                    
                    if len(unique_history) < len(value):
                        logger.warning(f"🔄 Removed {len(value) - len(unique_history)} duplicate actions from history (total: {len(value)} → {len(unique_history)})")
                    node_specific_data[key] = serialize_complex_object(unique_history)
                else:
                    node_specific_data[key] = serialize_complex_object(value)
                
                if key == "execution_plan" and value:
                     node_specific_data["description"] = f"Brain created/updated plan with {len(value)} phases"
                    
            elif node_name in ["omni_hands", "execute_next_action"]:
                # Handle task execution updates
                if key == "execution_plan":
                    node_specific_data["execution_plan"] = serialize_complex_object(value)
                    node_specific_data["description"] = "Updated execution plan status"

                elif key == "executed_task_id":
                    node_specific_data["executed_task_id"] = value
                    
                    # Try to find task description from todo_list if available in same output
                    todo_list = node_output.get("todo_list", [])
                    task_desc = "Unknown Task"
                    for t in todo_list:
                        if t.get("id") == value or t.get("task_id") == value:
                            task_desc = t.get("description", "Unknown Task")
                            break
                    node_specific_data["task_description"] = task_desc
                    node_specific_data["description"] = f"Executed: {task_desc}"

                elif key == "task_status":
                     node_specific_data["task_status"] = value
                
                # Forward execution result to frontend
                elif key == "execution_result":
                     node_specific_data["execution_result"] = serialize_complex_object(value)
                     if isinstance(value, dict) and value.get("success"):
                          node_specific_data["description"] = "Action executed successfully"
                     elif isinstance(value, dict):
                          node_specific_data["description"] = f"Action failed: {value.get('error_message')}"

            # --- LEGACY NODES (Keep for backward compatibility) ---
            elif node_name == "parse_prompt" and key == "parsed_tasks":
                if value:
                    # Handle both Task objects and dictionaries
                    task_names = []
                    logger.debug(f"Processing {len(value)} parsed tasks for thread_id {thread_id}")
                    for i, task in enumerate(value):
                        if hasattr(task, 'task_name'):
                            task_name = task.task_name
                            task_names.append(task_name)
                            logger.debug(f"Task {i}: {task_name} (Task object)")
                        elif isinstance(task, dict) and 'task_name' in task:
                            task_name = task['task_name']
                            task_names.append(task_name)
                            logger.debug(f"Task {i}: {task_name} (dict)")
                        else:
                            task_str = str(task)
                            task_names.append(task_str)
                            logger.debug(f"Task {i}: {task_str} (fallback)")

                        # Check for empty task names
                        if not task_names[-1] or task_names[-1].strip() == '':
                            logger.warning(f"Empty task name detected at index {i} for thread_id {thread_id}")

                    node_specific_data["tasks_identified"] = len(value)
                    node_specific_data["task_names"] = task_names
                    node_specific_data["description"] = "Identified and parsed user tasks"

                    # Filter out empty task names for logging
                    non_empty_tasks = [name for name in task_names if name and name.strip()]
                    logger.info(f"Successfully parsed {len(non_empty_tasks)} non-empty tasks: {non_empty_tasks}")
                else:
                    node_specific_data["tasks_identified"] = 0
                    node_specific_data["task_names"] = []
                    node_specific_data["description"] = "No tasks identified from prompt"
                    logger.warning(f"No tasks were parsed from prompt for thread_id {thread_id}")

            elif node_name == "agent_directory_search" and key == "candidate_agents":
                node_specific_data["agents_found"] = sum(len(agents) for agents in value.values()) if value else 0
                node_specific_data["tasks_with_agents"] = list(value.keys()) if value else []
                node_specific_data["description"] = "Found candidate agents for tasks"

            elif node_name == "rank_agents" and key == "task_agent_pairs":
                node_specific_data["pairs_created"] = len(value) if value else 0
                node_specific_data["description"] = "Ranked and paired agents with tasks"
                if value:
                    pairs_data = [serialize_complex_object(pair) for pair in value]
                    node_specific_data["task_agent_pairs"] = pairs_data

            elif node_name == "plan_execution" and key == "task_plan":
                node_specific_data["execution_plan_ready"] = True
                node_specific_data["planned_tasks"] = len(value) if value else 0
                node_specific_data["description"] = "Created execution plan"

            elif node_name == "execute_batch" and key in ["completed_tasks", "final_response"]:
                if key == "completed_tasks":
                    node_specific_data["tasks_completed"] = len(value) if value else 0
                    node_specific_data["description"] = "Executed task batch"
                elif key == "final_response":
                    node_specific_data["has_final_response"] = bool(value)
    else:
        serializable_data = {"raw_output": str(node_output)}
        node_specific_data["description"] = f"Node {node_name} completed"

    return {**serializable_data, **node_specific_data}

# --- API Endpoints ---
@app.post("/api/orchestrator/action/approve")
async def approve_action_endpoint(request: ActionApprovalRequest):
    """Approve a pending action and resume orchestration."""
    thread_id = request.thread_id
    logger.info(f"👍 APPROVING action for thread {thread_id}")

    try:
        from backend.orchestrator.omni_dispatcher import approve_pending_action
        
        # Get current state
        with store_lock:
            state = conversation_store.get(thread_id)
            
        if not state:
            raise HTTPException(status_code=404, detail="Conversation not found")
            
        if not state.get("pending_approval"):
            return {"status": "no_action_pending", "message": "No action is currently awaiting approval"}

        # Apply approval updates
        updates = approve_pending_action(state)
        new_state = {**state, **updates}
        
        # Update store
        with store_lock:
            conversation_store[thread_id] = new_state
            
        # Resume orchestration in background
        # Note: In a real implementation, we would trigger the graph again
        # For now, we update the state so the next 'continue' call works
        return {"status": "approved", "message": "Action approved. Please click 'Continue' to resume."}

    except Exception as e:
        logger.error(f"Failed to approve action: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/orchestrator/action/reject")
async def reject_action_endpoint(request: ActionRejectRequest):
    """Reject a pending action."""
    thread_id = request.thread_id
    logger.info(f"👎 REJECTING action for thread {thread_id}: {request.reason}")

    try:
        from backend.orchestrator.omni_dispatcher import reject_pending_action
        
        # Get current state
        with store_lock:
            state = conversation_store.get(thread_id)
            
        if not state:
            raise HTTPException(status_code=404, detail="Conversation not found")
            
        # Apply rejection updates
        updates = reject_pending_action(state, request.reason)
        new_state = {**state, **updates}
        
        # Update store
        with store_lock:
            conversation_store[thread_id] = new_state
            
        return {"status": "rejected", "message": "Action rejected and skipped."}

    except Exception as e:
        logger.error(f"Failed to reject action: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat", response_model=ProcessResponse)
async def find_agents(request: ProcessRequest, api_request: Request):
    """
    Receives a prompt, runs it through the agent-finding graph,
    and returns the selected primary and fallback agents for each task.
    Supports interactive workflows that may require user input.
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    logger.info(f"Starting agent search with thread_id: {thread_id}")
    
    # 🔴 RECORD TO MEGA LOGGER -> THE CORE USER REQUEST
    mega_log_block(
        logger, 
        f"API CHAT REQUEST | Thread: {thread_id}", 
        f"Prompt: {request.prompt}\nFiles Count: {len(request.files) if request.files else 0}"
    )

    # Get user auth
    owner = None
    try:
        from auth import get_user_from_request
        user = get_user_from_request(api_request)
        if user:
             user_id = user.get("sub") or user.get("user_id") or user.get("id")
             if user_id:
                 owner = {"user_id": user_id}
    except Exception as e:
        logger.warning(f"Could not extract user from request: {e}")

    try:
        final_state = await execute_orchestration(
            prompt=request.prompt,
            thread_id=thread_id,
            files=request.files,  # Pass the files to the orchestrator
            stream_callback=None,
            owner=owner
        )

        # Check if workflow is paused for user input
        if final_state.get("pending_user_input"):
            logger.info(f"Workflow paused for user input in thread_id: {thread_id}")
            return ProcessResponse(
                message="Additional information required to complete your request.",
                thread_id=thread_id,
                task_agent_pairs=[],
                final_response=None,
                pending_user_input=True,
                question_for_user=final_state.get("question_for_user")
            )

        task_agent_pairs = final_state.get("task_agent_pairs", [])
        final_response_str = final_state.get("final_response")

        # Check for a valid outcome
        if not task_agent_pairs and not final_response_str and not final_state.get("pending_user_input"):
            logger.warning(f"Could not parse any tasks or generate a response for thread_id: {thread_id}. Prompt: '{request.prompt}'")
            raise HTTPException(
                status_code=404,
                detail=f"I couldn't identify any specific tasks or generate a response from your message: '{request.prompt}'. Could you please be more specific?"
            )

        logger.info(f"Successfully processed request for thread_id: {thread_id}")

        # Get Canvas Registry state for this thread
        from backend.services.canvas_registry import get_canvas_registry
        canvas_registry = get_canvas_registry(thread_id)

        # Also merge any live canvas updates (e.g., from browser agent WebSocket)
        with canvas_lock:
            if thread_id in live_canvas_updates:
                live_data = live_canvas_updates[thread_id]
                if live_data.get('has_canvas'):
                    canvas_registry.register_sync(
                        canvas_id=f"live_{thread_id}_{int(time.time())}",
                        canvas_type=live_data.get('canvas_type', 'html'),
                        source_agent="browser_agent",
                        canvas_data=live_data.get('canvas_data'),
                        canvas_content=live_data.get('canvas_content'),
                    )
                logger.info(f"📊 Merged live canvas data for thread {thread_id}")

        # Build registry state and backward-compat fields
        registry_state = canvas_registry.get_registry_state()
        compat = canvas_registry.get_backward_compat_fields()

        # Also check final_state for registry (from Hands)
        state_registry = final_state.get('canvas_registry')
        state_active = final_state.get('active_canvas_id')

        return ProcessResponse(
            message="Successfully processed the request.",
            thread_id=thread_id,
            task_agent_pairs=task_agent_pairs,
            final_response=final_response_str,
            pending_user_input=False,
            question_for_user=None,
            # Canvas Registry (NEW)
            canvas_registry=registry_state if registry_state.canvases else None,
            active_canvas_id=canvas_registry.get_active_id() or state_active,
            # Backward compat fields
            has_canvas=compat.get('has_canvas') or final_state.get('has_canvas', False),
            canvas_content=compat.get('canvas_content') or final_state.get('canvas_content'),
            canvas_type=compat.get('canvas_type') or final_state.get('canvas_type'),
            canvas_data=compat.get('canvas_data') or final_state.get('canvas_data'),
            browser_view=compat.get('browser_view'),
            plan_view=compat.get('plan_view'),
            current_view=compat.get('current_view', 'browser'),
            # Omni-Dispatcher fields
            execution_plan=final_state.get('execution_plan'),
            action_history=final_state.get('action_history'),
            insights=final_state.get('insights'),
            pending_approval=final_state.get('pending_approval', False),
            pending_decision=final_state.get('pending_decision')
        )

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions directly to preserve the status code
        raise http_exc
    except Exception as e:
        logger.error(f"An unexpected error occurred during graph execution for thread_id {thread_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

@app.post("/api/chat/continue", response_model=ProcessResponse)
async def continue_conversation(user_response: UserResponse, api_request: Request):
    """
    Continue a paused conversation by providing user response to a question.
    """
    logger.info(f"Continuing conversation for thread_id: {user_response.thread_id} with response: {user_response.response[:100]}...")
    
    # 🔴 RECORD TO MEGA LOGGER -> THE CORE USER RESPONSE
    mega_log_block(
        logger, 
        f"API CHAT CONTINUE | Thread: {user_response.thread_id}", 
        f"User Response: {user_response.response}\nFiles Count: {len(user_response.files) if user_response.files else 0}"
    )

    # Check if conversation exists
    with store_lock:
        existing_conversation = conversation_store.get(user_response.thread_id)

    if not existing_conversation:
        logger.warning(f"No existing conversation found for thread_id: {user_response.thread_id}")
        raise HTTPException(status_code=404, detail="Conversation thread not found. Please start a new conversation.")

    logger.info(f"Found existing conversation for thread_id: {user_response.thread_id}, pending_input: {existing_conversation.get('pending_user_input', False)}")

    # Get user auth
    owner = None
    try:
        from auth import get_user_from_request
        user = get_user_from_request(api_request)
        if user:
             user_id = user.get("sub") or user.get("user_id") or user.get("id")
             if user_id:
                 owner = {"user_id": user_id}
    except Exception as e:
        logger.warning(f"Could not extract user from request: {e}")

    try:
        final_state = await execute_orchestration(
            prompt=None, # No new prompt needed
            thread_id=user_response.thread_id,
            user_response=user_response.response,
            files=user_response.files,  # Pass files if provided
            stream_callback=None,
            owner=owner
        )

        # Check if workflow is paused again for more user input
        if final_state.get("pending_user_input"):
            logger.info(f"Workflow paused again for user input in thread_id: {user_response.thread_id}")
            return ProcessResponse(
                message="Additional information required to complete your request.",
                thread_id=user_response.thread_id,
                task_agent_pairs=[],
                final_response=None,
                pending_user_input=True,
                question_for_user=final_state.get("question_for_user")
            )

        task_agent_pairs = final_state.get("task_agent_pairs", [])
        final_response_str = final_state.get("final_response")

        logger.info(f"Successfully continued conversation for thread_id: {user_response.thread_id}")

        # Get Canvas Registry state
        from backend.services.canvas_registry import get_canvas_registry
        canvas_registry = get_canvas_registry(user_response.thread_id)

        with canvas_lock:
            if user_response.thread_id in live_canvas_updates:
                live_data = live_canvas_updates[user_response.thread_id]
                if live_data.get('has_canvas'):
                    canvas_registry.register_sync(
                        canvas_id=f"live_{user_response.thread_id}_{int(time.time())}",
                        canvas_type=live_data.get('canvas_type', 'html'),
                        source_agent="browser_agent",
                        canvas_data=live_data.get('canvas_data'),
                        canvas_content=live_data.get('canvas_content'),
                    )

        registry_state = canvas_registry.get_registry_state()
        compat = canvas_registry.get_backward_compat_fields()

        return ProcessResponse(
            message="Successfully processed the continued conversation.",
            thread_id=user_response.thread_id,
            task_agent_pairs=task_agent_pairs,
            final_response=final_response_str,
            pending_user_input=False,
            question_for_user=None,
            canvas_registry=registry_state if registry_state.canvases else None,
            active_canvas_id=canvas_registry.get_active_id(),
            has_canvas=compat.get('has_canvas', False),
            canvas_content=compat.get('canvas_content'),
            canvas_type=compat.get('canvas_type'),
            canvas_data=compat.get('canvas_data'),
            browser_view=compat.get('browser_view'),
            plan_view=compat.get('plan_view'),
            current_view=compat.get('current_view', 'browser')
        )

    except Exception as e:
        logger.error(f"An unexpected error occurred during conversation continuation for thread_id {user_response.thread_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

# ============================================================================
# CANVAS REGISTRY API ENDPOINTS
# ============================================================================

class CanvasFocusRequest(BaseModel):
    thread_id: str
    canvas_id: str

class CanvasDismissRequest(BaseModel):
    thread_id: str
    canvas_id: str

@app.get("/api/canvas/templates", tags=["Canvas"])
async def list_canvas_templates(category: str = None, agent: str = None):
    """List all predefined canvas templates with their data schemas."""
    from services.canvas_templates import list_templates
    templates = list_templates(category=category, agent=agent)
    return {"templates": templates, "count": len(templates)}

@app.get("/api/canvas/{thread_id}", tags=["Canvas"])
async def get_canvas_registry_state(thread_id: str):
    """Get the full Canvas Registry state for a thread."""
    from backend.services.canvas_registry import get_canvas_registry
    registry = get_canvas_registry(thread_id)
    state = registry.get_registry_state()
    return {
        "thread_id": thread_id,
        "registry": state.model_dump(),
        "backward_compat": registry.get_backward_compat_fields(),
    }

@app.post("/api/canvas/focus", tags=["Canvas"])
async def set_canvas_focus(request: CanvasFocusRequest):
    """Set the active (focused) canvas for a thread."""
    from backend.services.canvas_registry import get_canvas_registry
    registry = get_canvas_registry(request.thread_id)
    success = await registry.set_active(request.canvas_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Canvas '{request.canvas_id}' not found or not active")
    return {
        "success": True,
        "active_canvas_id": request.canvas_id,
        "registry": registry.get_registry_state().model_dump(),
    }

@app.post("/api/canvas/dismiss", tags=["Canvas"])
async def dismiss_canvas(request: CanvasDismissRequest):
    """Dismiss a canvas (hide it without deleting)."""
    from backend.services.canvas_registry import get_canvas_registry
    registry = get_canvas_registry(request.thread_id)
    success = await registry.dismiss(request.canvas_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Canvas '{request.canvas_id}' not found")
    return {
        "success": True,
        "dismissed_canvas_id": request.canvas_id,
        "registry": registry.get_registry_state().model_dump(),
    }

# ============================================================================
# WEBSOCKET SCREENSHOT RELAY ENDPOINT
# ============================================================================
# Browser agent connects here to stream screenshots directly to frontend
# This eliminates HTTP overhead and asyncio polling issues
# ============================================================================

@app.websocket("/ws/screenshots/{thread_id}")
async def screenshots_websocket(websocket: WebSocket, thread_id: str):
    """
    WebSocket endpoint for browser agent to stream screenshots.
    Screenshots are immediately relayed to the frontend WebSocket for this thread_id.
    """
    await websocket.accept()
    logger.info(f"📸 Browser agent connected for screenshot streaming: thread={thread_id}")
    
    try:
        while True:
            # Receive screenshot data from browser agent
            data = await websocket.receive_json()
            
            screenshot_data = data.get("screenshot_data", "")
            url = data.get("url", "")
            step = data.get("step", 0)
            task_plan = data.get("task_plan", [])
            current_action = data.get("current_action", "")
            
            if not screenshot_data:
                continue
            
            # Create browser view HTML
            browser_view_html = f'''
            <div style="text-align: center;">
                <img src="data:image/jpeg;base64,{screenshot_data}" alt="Browser live view" style="width: 100%; max-width: 1200px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);" />
                <div style="margin-top: 10px; color: #666; font-size: 14px;">
                    <strong>🔴 Live Browser View</strong> | Step {step} | {url[:60] if url else 'Loading...'}
                </div>
            </div>
            '''
            
            # Relay to frontend WebSocket if connected
            async with frontend_ws_lock:
                logger.info(f"🔍 Looking for frontend WS: thread={thread_id}, registered={list(frontend_websockets.keys())}")
                frontend_ws = frontend_websockets.get(thread_id)
                if frontend_ws:
                    try:
                        await frontend_ws.send_json({
                            "node": "__live_canvas__",
                            "thread_id": thread_id,
                            "data": {
                                "has_canvas": True,
                                "canvas_type": "html",
                                "canvas_content": browser_view_html,
                                "browser_view": browser_view_html,
                                "current_view": "browser",
                                "screenshot_count": step
                            },
                            "timestamp": time.time()
                        })
                        logger.info(f"📡 Screenshot relayed to frontend: thread={thread_id}, step={step}, size={len(screenshot_data)}")
                    except Exception as e:
                        logger.warning(f"Failed to relay screenshot to frontend: {e}")
                        # Remove disconnected frontend
                        del frontend_websockets[thread_id]
                else:
                    logger.warning(f"⚠️ No frontend connected for thread={thread_id}, screenshot dropped")
                    
    except WebSocketDisconnect:
        logger.info(f"📸 Browser agent disconnected: thread={thread_id}")
    except Exception as e:
        logger.error(f"Error in screenshot WebSocket: {e}")


# NOTE: Conversation routes (/api/chat/*, /api/conversations/*) are defined in routers/conversations_router.py
# and included via app.include_router(conversations_router.router)

# NOTE: Workflow routes (/api/workflows/*, /api/schedules/*, /api/plan/*) are defined in routers/workflows_router.py
# and included via app.include_router(workflows_router.router)
@app.post("/api/workflows", tags=["Workflows"])
async def save_workflow(request: Request, thread_id: str, name: str, description: str = "", db: Session = Depends(get_db)):
    """Save conversation as reusable workflow"""
    from auth import get_user_from_request
    
    user = get_user_from_request(request)
    user_id = user.get("sub") or user.get("user_id") or user.get("id")
    
    # Load conversation
    history_path = os.path.join(CONVERSATION_HISTORY_DIR, f"{thread_id}.json")
    if not os.path.exists(history_path):
        fallback_dir = os.path.join(BACKEND_DIR, "agent_conversations")
        fallback_path = os.path.join(fallback_dir, f"{thread_id}.json")
        if os.path.exists(fallback_path):
            history_path = fallback_path
        else:
            raise HTTPException(status_code=404, detail="Conversation not found")
    
    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)
    
    workflow_id = str(uuid.uuid4())
    
    # The conversation history file structure is flat, not nested under 'state'
    # Extract data from both top-level and metadata locations
    
    # Check if we have the data directly in history (top-level - preferred)
    task_agent_pairs = history.get("task_agent_pairs", [])
    task_plan = history.get("task_plan", []) or history.get("plan", [])  # Fallback to "plan"
    original_prompt = history.get("original_prompt", "")
    todo_list = history.get("todo_list", [])
    
    # If data is missing at top-level, try metadata (fallback)
    if not task_agent_pairs or not task_plan or not original_prompt:
        metadata = history.get("metadata", {})
        task_agent_pairs = task_agent_pairs or metadata.get("task_agent_pairs", [])
        task_plan = task_plan or metadata.get("task_plan", []) or metadata.get("plan", [])
        original_prompt = original_prompt or metadata.get("original_prompt", "")
        logger.info(f"📋 Extracted from metadata: task_agent_pairs={len(task_agent_pairs)}, task_plan={len(task_plan)}")
    
    # If original_prompt is still missing, extract from first user message
    if not original_prompt:
        messages = history.get("messages", [])
        if messages and len(messages) > 0:
            first_msg = messages[0]
            if isinstance(first_msg, dict) and first_msg.get("type") in ["user", "human"]:
                original_prompt = first_msg.get("content", "")
                logger.info(f"Extracted original_prompt from first message: '{original_prompt[:50]}...'")
    
    # If still empty after fallbacks, try from checkpointer
    if not task_agent_pairs or not original_prompt or not todo_list:
        logger.info(f"History file missing data, attempting to load from checkpointer for {thread_id}")
        try:
            config = {"configurable": {"thread_id": thread_id}}
            checkpoint = checkpointer.get(config)
            if checkpoint:
                checkpoint_state = checkpoint.get("values", {})
                task_agent_pairs = checkpoint_state.get("task_agent_pairs", task_agent_pairs)
                task_plan = checkpoint_state.get("task_plan", task_plan)
                original_prompt = checkpoint_state.get("original_prompt", original_prompt)
                todo_list = checkpoint_state.get("todo_list", todo_list)
                # Also get other fields from checkpoint
                history = {**history, **checkpoint_state}
                logger.info("✅ Retrieved workflow data from checkpointer")
        except Exception as e:
            logger.warning(f"Could not load from checkpointer: {e}")
    
    # Extract comprehensive blueprint from conversation state
    task_plan = history.get("task_plan", []) or history.get("plan", [])
    
    # Log what we're putting in the blueprint for debugging
    logger.info(f"📋 Blueprint construction: task_agent_pairs_count={len(task_agent_pairs)}, task_plan_count={len(task_plan)}, prompt_len={len(original_prompt)}")
    
    blueprint = {
        "workflow_id": workflow_id,
        "thread_id": thread_id,
        "original_prompt": original_prompt,
        "task_agent_pairs": task_agent_pairs,
        "task_plan": task_plan,
        "todo_list": todo_list,
        "parsed_tasks": history.get("parsed_tasks", []),
        "candidate_agents": history.get("candidate_agents", {}),
        "user_expectations": history.get("user_expectations"),
        "completed_tasks": history.get("completed_tasks", []),
        "final_response": history.get("final_response"),
        "created_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    }
    
    # Generate plan_graph structure from task_plan for visualization
    # The frontend PlanGraph component can reconstruct this, but we store it for convenience
    plan_graph = None
    if task_plan:
        plan_graph = {
            "nodes": [],
            "edges": [],
            "tasks": task_plan
        }
    
    workflow = Workflow(
        workflow_id=workflow_id,
        user_id=user_id,
        name=name,
        description=description or blueprint.get("original_prompt", "")[:200],  # Use prompt as fallback description
        blueprint=blueprint,
        plan_graph=plan_graph
    )
    db.add(workflow)
    db.commit()
    
    logger.info(f"Workflow '{name}' ({workflow_id}) saved by user {user_id}")
    return {
        "workflow_id": workflow_id,
        "name": name,
        "description": workflow.description,
        "status": "saved",
        "task_count": len(blueprint.get("task_agent_pairs", [])),
        "created_at": blueprint["created_at"]
    }

@app.get("/api/workflows", tags=["Workflows"])
async def list_workflows(request: Request, db: Session = Depends(get_db)):
    """List user's workflows"""
    from auth import get_user_from_request
    
    user = get_user_from_request(request)
    user_id = user.get("sub") or user.get("user_id") or user.get("id")
    
    workflows = db.query(Workflow).filter_by(user_id=user_id, status='active').all()
    
    result = []
    for w in workflows:
        # Extract task count from blueprint
        blueprint = w.blueprint or {}
        task_count = len(blueprint.get("task_agent_pairs", []))
        
        # Calculate estimated cost from completed tasks if available
        estimated_cost = 0.0
        completed_tasks = blueprint.get("completed_tasks", [])
        for task in completed_tasks:
            if isinstance(task, dict):
                estimated_cost += task.get("cost", 0.0)
        
        # Get active schedules for this workflow
        from models import WorkflowSchedule
        active_schedules = db.query(WorkflowSchedule).filter_by(
            workflow_id=w.workflow_id, 
            is_active=True
        ).all()
        schedule_count = len(active_schedules)
        
        # Get next scheduled run time
        next_scheduled_run = None
        if active_schedules:
            next_schedule = min(active_schedules, key=lambda s: s.next_run_at or datetime.now(timezone.utc).replace(tzinfo=None))
            if next_schedule.next_run_at:
                next_scheduled_run = next_schedule.next_run_at.isoformat()
        
        result.append({
            "workflow_id": w.workflow_id,
            "workflow_name": w.name,
            "workflow_description": w.description or "No description",
            "created_at": w.created_at.isoformat(),
            "updated_at": w.updated_at.isoformat() if w.updated_at else w.created_at.isoformat(),
            "task_count": task_count,
            "estimated_cost": estimated_cost,
            "has_plan_graph": w.plan_graph is not None,
            "is_public": False,  # Add public flag support later
            "active_schedules": schedule_count,
            "next_scheduled_run": next_scheduled_run
        })
    
    return result

@app.get("/api/workflows/{workflow_id}", tags=["Workflows"])
async def get_workflow(workflow_id: str, request: Request, db: Session = Depends(get_db)):
    """Get workflow details"""
    from auth import get_user_from_request
    
    user = get_user_from_request(request)
    user_id = user.get("sub") or user.get("user_id") or user.get("id")
    
    workflow = db.query(Workflow).filter_by(workflow_id=workflow_id, user_id=user_id).first()
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    return {
        "workflow_id": workflow.workflow_id,
        "name": workflow.name,
        "description": workflow.description,
        "blueprint": workflow.blueprint,
        "plan_graph": workflow.plan_graph,
        "created_at": workflow.created_at.isoformat()
    }

@app.delete("/api/workflows/{workflow_id}", tags=["Workflows"])
async def delete_workflow(workflow_id: str, request: Request, db: Session = Depends(get_db)):
    """Delete a saved workflow and its associated schedules"""
    from auth import get_user_from_request

    user = get_user_from_request(request)
    user_id = user.get("sub") or user.get("user_id") or user.get("id")

    workflow = db.query(Workflow).filter_by(workflow_id=workflow_id, user_id=user_id).first()
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    # Cancel any active schedules for this workflow
    try:
        from backend.services.workflow_scheduler import get_scheduler
        scheduler = get_scheduler()
        schedules = db.query(WorkflowSchedule).filter_by(workflow_id=workflow_id).all()
        for schedule in schedules:
            try:
                scheduler.remove_schedule(str(schedule.schedule_id))
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"Could not cancel schedules for workflow {workflow_id}: {e}")

    db.delete(workflow)
    db.commit()
    logger.info(f"Workflow {workflow_id} deleted by user {user_id}")
    return {"status": "deleted", "workflow_id": workflow_id}


@app.post("/api/workflows/{workflow_id}/clone", tags=["Workflows"])
async def clone_workflow(workflow_id: str, request: Request, body: Dict[str, Any] = Body(default={}), db: Session = Depends(get_db)):
    """Clone an existing workflow with an optional new name"""
    from auth import get_user_from_request

    user = get_user_from_request(request)
    user_id = user.get("sub") or user.get("user_id") or user.get("id")

    source = db.query(Workflow).filter_by(workflow_id=workflow_id, user_id=user_id).first()
    if not source:
        raise HTTPException(status_code=404, detail="Workflow not found")

    new_id = str(uuid.uuid4())
    new_name = body.get("new_name") or f"Copy of {source.name}"

    clone = Workflow(
        workflow_id=new_id,
        user_id=user_id,
        name=new_name,
        description=source.description,
        blueprint={**(source.blueprint or {}), "workflow_id": new_id},
        plan_graph=source.plan_graph,
    )
    db.add(clone)
    db.commit()
    logger.info(f"Workflow {workflow_id} cloned as {new_id} by user {user_id}")
    return {
        "workflow_id": new_id,
        "name": new_name,
        "description": clone.description,
        "status": "cloned",
    }


@app.post("/api/workflows/{workflow_id}/execute", tags=["Workflows"])
async def execute_workflow(workflow_id: str, request: Request, inputs: Dict[str, Any] = Body(default={}), db: Session = Depends(get_db)):
    """Execute workflow by re-running the saved plan (no re-planning)"""
    from auth import get_user_from_request
    from langgraph.checkpoint.memory import MemorySaver
    
    user = get_user_from_request(request)
    user_id = user.get("sub") or user.get("user_id") or user.get("id")
    
    workflow = db.query(Workflow).filter_by(workflow_id=workflow_id, user_id=user_id).first()
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    execution_id = str(uuid.uuid4())
    new_thread_id = str(uuid.uuid4())
    
    # Create execution record
    execution = WorkflowExecution(
        execution_id=execution_id,
        workflow_id=workflow_id,
        user_id=user_id,
        inputs=inputs,
        status='running'
    )
    db.add(execution)
    db.commit()
    
    # Get the saved plan from blueprint
    blueprint = workflow.blueprint
    task_plan = blueprint.get("task_plan", [])
    task_agent_pairs = blueprint.get("task_agent_pairs", [])
    original_prompt = blueprint.get("original_prompt", "")
    
    if not task_plan or not task_agent_pairs:
        raise HTTPException(status_code=400, detail="Workflow has no saved execution plan")
    
    # Create a memory saver for this thread
    memory = MemorySaver()
    
    # Pre-seed the thread state with the saved plan (bypass planning phase)
    initial_state = {
        "thread_id": new_thread_id,
        "original_prompt": original_prompt,
        "task_plan": task_plan,
        "task_agent_pairs": task_agent_pairs,
        "messages": [{"role": "user", "content": original_prompt}],
        "status": "executing",
        "planning_mode": False,
        "plan_approved": True,  # Skip approval, go straight to execution
        "completed_tasks": [],
        "task_events": []
    }
    
    # Save initial state to memory so WebSocket can pick it up
    config = {"configurable": {"thread_id": new_thread_id}}
    memory.put(config, {
        "values": initial_state,
        "next": ["execute_batch"]  # Start directly at execution
    })
    
    logger.info(f"Pre-seeded workflow execution {workflow_id} as thread {new_thread_id} with {len(task_plan)} batches")
    return {
        "execution_id": execution_id, 
        "thread_id": new_thread_id, 
        "status": "running", 
        "task_count": len(task_agent_pairs),
        "message": "Connect to /ws/chat with this thread_id - execution will start automatically"
    }

@app.post("/api/workflows/{workflow_id}/create-conversation", tags=["Workflows"])
async def create_workflow_conversation(workflow_id: str, request: Request, db: Session = Depends(get_db)):
    """
    Create a new conversation pre-seeded with the workflow's saved plan.
    This endpoint creates a new thread with the plan already loaded, allowing
    the user to review and modify it before execution.
    
    Returns thread_id and the plan details (task_agent_pairs, task_plan, original_prompt)
    """
    from auth import get_user_from_request
    from models import UserThread
    
    user = get_user_from_request(request)
    user_id = user.get("sub") or user.get("user_id") or user.get("id")
    logger.info(f"create_workflow_conversation: user_id={user_id}")
    
    if not user_id:
        raise HTTPException(status_code=401, detail="Unable to determine user identity")
    
    workflow = db.query(Workflow).filter_by(workflow_id=workflow_id, user_id=user_id).first()
    if not workflow:
        logger.error(f"Workflow {workflow_id} not found for user {user_id}")
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    # Create a new thread
    new_thread_id = str(uuid.uuid4())
    
    # Get the saved plan from blueprint
    blueprint = workflow.blueprint
    
    # Handle blueprint stored as JSON string
    if isinstance(blueprint, str):
        try:
            blueprint = json.loads(blueprint)
            logger.info("Deserialized blueprint from JSON string")
        except Exception as e:
            logger.error(f"Failed to deserialize blueprint JSON: {e}")
            raise HTTPException(status_code=400, detail="Invalid workflow blueprint format")
    
    task_plan = blueprint.get("task_plan", [])
    task_agent_pairs = blueprint.get("task_agent_pairs", [])
    original_prompt = blueprint.get("original_prompt", "")
    
    logger.info(f"Retrieved blueprint with {len(task_agent_pairs)} task pairs and {len(task_plan)} plan items")
    logger.info(f"Blueprint keys: {list(blueprint.keys())}")
    logger.info(f"Blueprint task_agent_pairs type: {type(task_agent_pairs)}, value: {task_agent_pairs}")
    logger.info(f"Blueprint task_plan type: {type(task_plan)}, value: {task_plan}")
    
    # We need task_agent_pairs to proceed - task_plan can be empty (will be generated if needed)
    if not task_agent_pairs:
        raise HTTPException(status_code=400, detail=f"Workflow has no task agent pairs. task_agent_pairs count: {len(task_agent_pairs) if task_agent_pairs else 0}, blueprint keys: {list(blueprint.keys())}")
    
    # Create UserThread record in database
    logger.info(f"Creating UserThread: thread_id={new_thread_id}, user_id={user_id}, workflow_name={workflow.name}")
    user_thread = UserThread(
        thread_id=new_thread_id,
        user_id=user_id,
        title=f"From: {workflow.name}"
    )
    db.add(user_thread)
    db.commit()
    logger.info("UserThread created successfully")
    
    # Save the conversation JSON file so it persists and frontend can load it
    # This pre-seeds the plan so it displays immediately in the sidebar
    history_path = os.path.join(CONVERSATION_HISTORY_DIR, f"{new_thread_id}.json")
    try:
        os.makedirs(CONVERSATION_HISTORY_DIR, exist_ok=True)
        conversation_json = {
            "thread_id": new_thread_id,
            "original_prompt": original_prompt,
            "task_agent_pairs": task_agent_pairs,
            "task_plan": task_plan,
            "messages": [],
            "completed_tasks": [],
            "final_response": None,
            "pending_user_input": True,
            "question_for_user": "Review the execution plan and approve to proceed.",
            "needs_approval": True,
            "approval_required": True,
            "plan_approved": False,
            "status": "planning_complete",
            "metadata": {
                "from_workflow": workflow_id,
                "workflow_name": workflow.name
            },
            "uploaded_files": []
        }
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(conversation_json, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved conversation JSON file for thread {new_thread_id} from workflow {workflow_id}")
    except Exception as e:
        logger.warning(f"Failed to save conversation JSON file: {e}")
        # Don't fail the whole operation if JSON save fails
    
    logger.info(f"Created new conversation {new_thread_id} from workflow {workflow_id} for user {user_id}")
    
    return {
        "thread_id": new_thread_id,
        "workflow_id": workflow_id,
        "original_prompt": original_prompt,
        "task_agent_pairs": task_agent_pairs,
        "task_plan": task_plan,
        "task_count": len(task_agent_pairs),
        "status": "planning_complete",
        "message": "Plan loaded. Review and run from the input box."
    }

@app.post("/api/workflows/{workflow_id}/schedule", tags=["Workflows"])
async def schedule_workflow(workflow_id: str, body: ScheduleWorkflowRequest, request: Request, db: Session = Depends(get_db)):
    """Schedule workflow execution with cron expression"""
    from auth import get_user_from_request
    from backend.services.workflow_scheduler import get_scheduler
    from database import SessionLocal
    
    user = get_user_from_request(request)
    user_id = user.get("sub")
    
    workflow = db.query(Workflow).filter_by(workflow_id=workflow_id, user_id=user_id).first()
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    # Check if workflow has a saved execution plan
    # Accept either task_plan (preferred) or task_agent_pairs (fallback)
    if not workflow.blueprint:
        raise HTTPException(
            status_code=400, 
            detail="Workflow has no blueprint data. Please save the workflow again to capture the execution plan."
        )
    
    has_task_plan = workflow.blueprint.get("task_plan") and len(workflow.blueprint.get("task_plan", [])) > 0
    has_task_agent_pairs = workflow.blueprint.get("task_agent_pairs") and len(workflow.blueprint.get("task_agent_pairs", [])) > 0
    
    if not has_task_plan and not has_task_agent_pairs:
        # Provide helpful error message with recovery options
        missing_fields = []
        if not has_task_plan:
            missing_fields.append("task_plan")
        if not has_task_agent_pairs:
            missing_fields.append("task_agent_pairs")
        
        raise HTTPException(
            status_code=400, 
            detail=f"Workflow has no execution plan. Missing: {', '.join(missing_fields)}. Recovery options: "
                   f"1) Complete a new conversation with the same prompt and save it again, "
                   f"2) Use a different workflow that has an execution plan, "
                   f"3) Run the workflow once to generate a plan before scheduling."
        )
    
    schedule_id = str(uuid.uuid4())
    schedule = WorkflowSchedule(
        schedule_id=schedule_id,
        workflow_id=workflow_id,
        user_id=user_id,
        cron_expression=body.cron_expression,  # Already in UTC format from frontend
        input_template=body.input_template
    )
    db.add(schedule)
    db.commit()
    
    # Add to scheduler
    try:
        scheduler = get_scheduler()
        scheduler.add_schedule(
            schedule_id=schedule_id,
            workflow_id=workflow_id,
            cron_expression=body.cron_expression,
            input_template=body.input_template,
            user_id=user_id,
            db_session_factory=SessionLocal
        )
        logger.info(f"Scheduled workflow {workflow_id} with cron: {body.cron_expression}")
        return {"schedule_id": schedule_id, "status": "scheduled", "cron": body.cron_expression}
    except Exception as e:
        # Rollback if scheduling fails
        db.delete(schedule)
        db.commit()
        logger.error(f"Failed to schedule workflow: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid cron expression or scheduling error: {str(e)}")

@app.post("/api/workflows/{workflow_id}/webhook", tags=["Workflows"])
async def create_webhook(workflow_id: str, request: Request, db: Session = Depends(get_db)):
    """Create webhook trigger for workflow"""
    from auth import get_user_from_request
    import secrets
    
    user = get_user_from_request(request)
    user_id = user.get("sub")
    
    workflow = db.query(Workflow).filter_by(workflow_id=workflow_id, user_id=user_id).first()
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    webhook_id = str(uuid.uuid4())
    webhook_token = secrets.token_urlsafe(32)
    
    webhook = WorkflowWebhook(
        webhook_id=webhook_id,
        workflow_id=workflow_id,
        user_id=user_id,
        webhook_token=webhook_token
    )
    db.add(webhook)
    db.commit()
    
    logger.info(f"Created webhook {webhook_id} for workflow {workflow_id}")
    return {"webhook_id": webhook_id, "webhook_url": f"/webhooks/{webhook_id}", "webhook_token": webhook_token}

@app.get("/api/schedules", tags=["Workflows"])
async def list_schedules(request: Request, db: Session = Depends(get_db)):
    """List all workflow schedules for the authenticated user"""
    from auth import get_user_from_request
    
    user = get_user_from_request(request)
    user_id = user.get("sub")
    
    schedules = db.query(WorkflowSchedule).filter_by(user_id=user_id).order_by(WorkflowSchedule.created_at.desc()).all()
    
    result = []
    for schedule in schedules:
        workflow = db.query(Workflow).filter_by(workflow_id=schedule.workflow_id).first()
        
        # Calculate next run time from cron expression
        next_run = None
        if schedule.is_active:
            try:
                from apscheduler.triggers.cron import CronTrigger
                from datetime import timezone
                parts = schedule.cron_expression.split()
                if len(parts) == 5:
                    minute, hour, day, month, day_of_week = parts
                    trigger = CronTrigger(
                        minute=minute, hour=hour, day=day, 
                        month=month, day_of_week=day_of_week, timezone='UTC'
                    )
                    # Get next fire time after current UTC time
                    now = datetime.now(timezone.utc)
                    next_run = trigger.get_next_fire_time(None, now)
                    if next_run:
                        next_run = next_run.isoformat()
            except Exception as e:
                logger.error(f"Failed to calculate next run time: {str(e)}")
        
        result.append({
            "schedule_id": schedule.schedule_id,
            "workflow_id": schedule.workflow_id,
            "workflow_name": workflow.name if workflow else "Unknown",
            "cron_expression": schedule.cron_expression,
            "input_template": schedule.input_template,
            "is_active": schedule.is_active,
            "last_run_at": schedule.last_run_at.isoformat() if schedule.last_run_at else None,
            "next_run_at": next_run,
            "created_at": schedule.created_at.isoformat()
        })
    
    return {"schedules": result, "count": len(result)}

@app.get("/api/schedules/{schedule_id}/executions", tags=["Workflows"])
async def get_schedule_executions(schedule_id: str, request: Request, db: Session = Depends(get_db)):
    """Get execution history for a specific schedule"""
    from auth import get_user_from_request
    
    user = get_user_from_request(request)
    user_id = user.get("sub")
    
    schedule = db.query(WorkflowSchedule).filter_by(
        schedule_id=schedule_id,
        user_id=user_id
    ).first()
    
    if not schedule:
        raise HTTPException(status_code=404, detail="Schedule not found")
    
    # Get all executions for this workflow (we'll filter by time range near schedule runs)
    executions = db.query(WorkflowExecution).filter_by(
        workflow_id=schedule.workflow_id,
        user_id=user_id
    ).order_by(WorkflowExecution.started_at.desc()).limit(50).all()
    
    result = []
    for execution in executions:
        result.append({
            "execution_id": execution.execution_id,
            "status": execution.status,
            "inputs": execution.inputs,
            "outputs": execution.outputs,
            "error": execution.error,
            "started_at": execution.started_at.isoformat() if execution.started_at else None,
            "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
            "duration_ms": int((execution.completed_at - execution.started_at).total_seconds() * 1000) if execution.completed_at and execution.started_at else None
        })
    
    return {"executions": result, "count": len(result)}

@app.patch("/api/schedules/{schedule_id}", tags=["Workflows"])
async def update_schedule(schedule_id: str, body: UpdateScheduleRequest, request: Request, db: Session = Depends(get_db)):
    """Update a workflow schedule (pause/resume, change cron, update inputs)"""
    from auth import get_user_from_request
    from backend.services.workflow_scheduler import get_scheduler
    from database import SessionLocal
    
    user = get_user_from_request(request)
    user_id = user.get("sub")
    
    schedule = db.query(WorkflowSchedule).filter_by(
        schedule_id=schedule_id,
        user_id=user_id
    ).first()
    
    if not schedule:
        raise HTTPException(status_code=404, detail="Schedule not found")
    
    scheduler = get_scheduler()
    
    # Update fields
    if body.is_active is not None:
        schedule.is_active = body.is_active
        
        if body.is_active:
            # Re-add to scheduler
            try:
                scheduler.add_schedule(
                    schedule_id=schedule.schedule_id,
                    workflow_id=schedule.workflow_id,
                    cron_expression=schedule.cron_expression,
                    input_template=schedule.input_template or {},
                    user_id=schedule.user_id,
                    db_session_factory=SessionLocal
                )
                logger.info(f"Resumed schedule {schedule_id}")
            except Exception as e:
                logger.error(f"Failed to resume schedule: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Failed to resume schedule: {str(e)}")
        else:
            # Remove from scheduler
            try:
                scheduler.remove_schedule(schedule_id)
                logger.info(f"Paused schedule {schedule_id}")
            except Exception as e:
                logger.error(f"Failed to pause schedule: {str(e)}")
    
    if body.cron_expression is not None:
        schedule.cron_expression = body.cron_expression
        
        # Update in scheduler if active
        if schedule.is_active:
            try:
                scheduler.add_schedule(
                    schedule_id=schedule.schedule_id,
                    workflow_id=schedule.workflow_id,
                    cron_expression=body.cron_expression,
                    input_template=schedule.input_template or {},
                    user_id=schedule.user_id,
                    db_session_factory=SessionLocal
                )
                logger.info(f"Updated cron expression for schedule {schedule_id}")
            except Exception as e:
                logger.error(f"Failed to update schedule: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Invalid cron expression: {str(e)}")
    
    if body.input_template is not None:
        schedule.input_template = body.input_template
    
    db.commit()
    
    return {
        "schedule_id": schedule.schedule_id,
        "is_active": schedule.is_active,
        "cron_expression": schedule.cron_expression,
        "input_template": schedule.input_template,
        "status": "updated"
    }


@app.post("/api/admin/reload-schedules", tags=["Admin"])
async def reload_schedules(
    db: Session = Depends(get_db)
):
    """
    Reload all active schedules from database into the scheduler.
    Useful for debugging or recovering from scheduler initialization issues.
    """
    from backend.services.workflow_scheduler import get_scheduler
    scheduler = get_scheduler()
    
    try:
        # Load active schedules
        scheduler.load_active_schedules(db)
        
        # Get job count
        jobs = scheduler.scheduler.get_jobs()
        job_details = []
        
        for job in jobs:
            job_details.append({
                "job_id": job.id,
                "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None
            })
        
        logger.info(f"Reloaded schedules - {len(jobs)} jobs now active")
        
        return {
            "status": "success",
            "jobs_loaded": len(jobs),
            "jobs": job_details
        }
    except Exception as e:
        logger.error(f"Failed to reload schedules: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to reload schedules: {str(e)}")

@app.delete("/api/workflows/{workflow_id}/schedule/{schedule_id}", tags=["Workflows"])
async def delete_schedule(workflow_id: str, schedule_id: str, request: Request, db: Session = Depends(get_db)):
    """Delete a workflow schedule"""
    from auth import get_user_from_request
    from backend.services.workflow_scheduler import get_scheduler
    
    user = get_user_from_request(request)
    user_id = user.get("sub")
    
    schedule = db.query(WorkflowSchedule).filter_by(
        schedule_id=schedule_id,
        workflow_id=workflow_id,
        user_id=user_id
    ).first()
    
    if not schedule:
        raise HTTPException(status_code=404, detail="Schedule not found")
    
    # Remove from scheduler
    try:
        scheduler = get_scheduler()
        scheduler.remove_schedule(schedule_id)
    except Exception as e:
        logger.error(f"Failed to remove schedule from scheduler: {str(e)}")
    
    # Delete from database
    db.delete(schedule)
    db.commit()
    
    logger.info(f"Deleted schedule {schedule_id} for workflow {workflow_id}")
    return {"status": "deleted"}

@app.post("/webhooks/{webhook_id}", tags=["Webhooks"])
async def trigger_webhook(webhook_id: str, payload: Dict[str, Any], webhook_token: str = Query(...), db: Session = Depends(get_db)):
    """Trigger workflow via webhook - executes asynchronously"""
    webhook = db.query(WorkflowWebhook).filter_by(webhook_id=webhook_id, is_active=True).first()
    
    if not webhook or webhook.webhook_token != webhook_token:
        raise HTTPException(status_code=404, detail="Invalid webhook")
    
    workflow = db.query(Workflow).filter_by(workflow_id=webhook.workflow_id).first()
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    execution_id = str(uuid.uuid4())
    execution = WorkflowExecution(
        execution_id=execution_id,
        workflow_id=webhook.workflow_id,
        user_id=webhook.user_id,
        inputs=payload,
        status='queued',
        started_at=datetime.now(timezone.utc).replace(tzinfo=None)
    )
    db.add(execution)
    db.commit()
    
    # Execute workflow in background
    from backend.services.workflow_scheduler import get_scheduler
    scheduler = get_scheduler()
    asyncio.create_task(
        scheduler._async_execute_workflow(
            execution_id, workflow.workflow_id, workflow.blueprint, payload, webhook.user_id
        )
    )
    
    logger.info(f"Webhook {webhook_id} triggered workflow {webhook.workflow_id} (execution: {execution_id})")
    return {"execution_id": execution_id, "status": "running", "message": "Workflow execution started"}

@app.get("/api/plan/{thread_id}", response_model=PlanResponse)
async def get_agent_plan(thread_id: str):
    """
    Retrieves the markdown execution plan for a given conversation thread.
    """
    # Check both possible locations for plan files
    plan_dirs = ["agent_plans", "backend/agent_plans"]  # Check root first, then backend/
    file_path = None
    
    for plan_dir in plan_dirs:
        temp_path = os.path.join(plan_dir, f"{thread_id}-plan.md")
        if os.path.exists(temp_path):
            file_path = temp_path
            break

    if not file_path:
        raise HTTPException(
            status_code=404,
            detail=f"Plan file not found for thread_id: {thread_id} in any location: {plan_dirs}"
        )

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return PlanResponse(thread_id=thread_id, content=content)
    except Exception as e:
        logger.error(f"Error reading plan file for thread_id {thread_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="An internal server error occurred while reading the plan file."
        )

@app.websocket("/ws/workflow/{workflow_id}/execute")
async def websocket_workflow_execute(websocket: WebSocket, workflow_id: str, token: str = None):
    """Execute workflow via WebSocket with streaming - uses WorkflowExecutor"""
    await websocket.accept()
    thread_id = None
    execution_id = None
    db = None
    
    try:
        # Note: WorkflowExecutor is deprecated - this endpoint may need refactoring
        from auth import verify_clerk_token
        
        # Verify token from query params
        if not token:
            await websocket.send_json({"node": "__error__", "error": "Missing authentication token"})
            return
        
        # verify_clerk_token expects "Bearer <token>" format
        user = verify_clerk_token(f"Bearer {token}")
        if not user:
            await websocket.send_json({"node": "__error__", "error": "Invalid authentication token"})
            return
        
        user_id = user.get("sub") or user.get("user_id") or user.get("id")
        
        data = await websocket.receive_json()
        inputs = data.get("inputs", {})
        
        # Create owner object for orchestrator
        owner = {"user_id": user_id, "sub": user_id}
        
        # Get workflow
        db = SessionLocal()
        workflow = db.query(Workflow).filter_by(workflow_id=workflow_id, user_id=user_id).first()
        
        if not workflow:
            await websocket.send_json({"node": "__error__", "error": "Workflow not found"})
            return
        
        # Create execution record
        execution_id = str(uuid.uuid4())
        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_id=workflow_id,
            user_id=user_id,
            inputs=inputs,
            status='running'
        )
        db.add(execution)
        db.commit()
        
        # TODO: Implement workflow execution logic
        # For now, this endpoint is not functional
        await websocket.send_json({
            "node": "__error__",
            "error": "Workflow WebSocket execution is currently not implemented"
        })
        return
        
    except Exception as e:
        logger.error(f"Workflow execution error: {e}", exc_info=True)
        await websocket.send_json({"node": "__error__", "error": str(e)})
    
    finally:
        if db:
            db.close()

def _sanitize_for_json(obj):
    """Recursively replace NaN/Infinity float values with None (JSON null).
    Python's json module serializes float('nan') as NaN which is invalid JSON."""
    import math
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


async def safe_websocket_send(websocket: WebSocket, data: dict, thread_id: str = "unknown"):
    """
    Safely send JSON data through WebSocket, handling closed connections gracefully.
    """
    try:
        await websocket.send_json(_sanitize_for_json(data))
        return True
    except Exception as e:
        # Connection is closed or broken - log as error to debug issues
        logger.error(f"WebSocket send failed for thread {thread_id}: {e}")
        return False

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for streaming agent orchestration updates.
    Uses the unified orchestration service with streaming enabled.
    Now supports interactive workflows with user input.
    Enhanced with comprehensive error handling and logging.
    """
    try:
        await websocket.accept()
        logger.info(f"WebSocket connection accepted from {websocket.client}")
    except Exception as e:
        logger.error(f"Failed to accept WebSocket connection: {e}")
        return
    
    thread_id = None  # Initialize thread_id to None
    
    try:
        while True:  # Keep the connection open for multiple messages
            # Wait for message from client
            try:
                data = await websocket.receive_json()
            except json.JSONDecodeError as je:
                # Invalid JSON from client
                logger.error(f"Invalid JSON received from WebSocket client: {je}")
                try:
                    await websocket.send_json({
                        "node": "__error__",
                        "error": "Invalid message format. Expected valid JSON.",
                        "error_type": "JSONDecodeError",
                        "timestamp": time.time()
                    })
                except Exception as send_err:
                    logger.error(f"Failed to send JSON error response: {send_err}")
                continue  # Wait for next message
            except Exception as e:
                # Connection closed or other receive error
                error_type = type(e).__name__
                # WebSocketDisconnect and RuntimeError(websocket.client_state) are expected on close
                if "closed" not in str(e).lower() and error_type != "WebSocketDisconnect":
                    logger.warning(f"WebSocket receive error ({error_type}): {e}")
                break  # Exit the loop, connection is closed

            # Get thread_id from client, or generate a new one if not provided
            thread_id = data.get("thread_id") or str(uuid.uuid4())
            prompt = data.get("prompt")
            user_response = data.get("user_response")  # For continuing conversations
            files_data = data.get("files", [])  # Get files from WebSocket message
            # Pass through owner info if frontend sends Clerk-verified identity
            owner = data.get("owner")  # Expected shape: { user_id, email } or legacy string user_id
            if isinstance(owner, str):
                owner = {"user_id": owner}
            elif owner and not isinstance(owner, dict):
                owner = None

            planning_mode = data.get("planning_mode", False)  # Get planning mode flag

            logger.info(f"WebSocket received message with thread_id: {thread_id}, planning_mode: {planning_mode}")
            logger.info(f"Message details: has_prompt={bool(prompt)}, has_user_response={bool(user_response)}, prompt_value='{prompt[:50] if prompt else None}', user_response_value='{user_response[:50] if user_response else None}'")
            logger.info(f"Owner info: owner={owner}, type={type(owner)}")

            # Determine if this is a new thread
            is_new_thread = "thread_id" not in data or not data.get("thread_id")
            
            # For new threads, owner is required
            if is_new_thread and not owner:
                await websocket.send_json({
                    "node": "__error__",
                    "error": "Owner information is required for new conversations",
                    "error_type": "ValidationError",
                    "thread_id": thread_id,
                    "timestamp": time.time()
                })
                continue  # Continue waiting for messages
            
            # For existing threads, if owner not provided, try to get from database
            if not is_new_thread and not owner:
                try:
                    from database import SessionLocal
                    from models import UserThread
                    
                    db = SessionLocal()
                    try:
                        user_thread = db.query(UserThread).filter_by(thread_id=thread_id).first()
                        if user_thread:
                            owner = {"user_id": user_thread.user_id}
                            logger.info(f"Retrieved owner from database for existing thread {thread_id}: {owner}")
                        else:
                            logger.warning(f"No user-thread relationship found for thread {thread_id}")
                    finally:
                        db.close()
                except Exception as db_err:
                    logger.error(f"Failed to retrieve owner for thread {thread_id}: {db_err}")
                    # Continue anyway - proceed without owner for existing thread

            # Handle canvas confirmation messages
            message_type = data.get("type")
            
            # Check if user typed a confirmation word while confirmation is pending
            if not message_type and prompt:
                # Check if there's a pending confirmation
                try:
                    from backend.orchestrator.graph import graph
                    config = {"configurable": {"thread_id": thread_id}}
                    current_state = graph.get_state(config)
                    
                    if current_state and current_state.values and current_state.values.get("pending_confirmation"):
                        # Check if the prompt is a confirmation word
                        confirmation_words = ["yes", "confirm", "ok", "okay", "sure", "proceed", "do it", "apply", "go ahead"]
                        cancel_words = ["no", "cancel", "abort", "stop", "don't", "nevermind"]
                        
                        prompt_lower = prompt.lower().strip()
                        
                        if any(word in prompt_lower for word in confirmation_words):
                            logger.info(f"📝 User typed confirmation: '{prompt}' - treating as canvas confirmation")
                            # Convert to canvas_confirmation message
                            message_type = "canvas_confirmation"
                            data["type"] = "canvas_confirmation"
                            data["action"] = "confirm"
                            pending_task = current_state.values.get("pending_confirmation_task")
                            data["task_name"] = pending_task.get("task_name") if pending_task else None
                        elif any(word in prompt_lower for word in cancel_words):
                            logger.info(f"📝 User typed cancellation: '{prompt}' - treating as canvas cancellation")
                            # Convert to canvas_confirmation message with cancel action
                            message_type = "canvas_confirmation"
                            data["type"] = "canvas_confirmation"
                            data["action"] = "cancel"
                            pending_task = current_state.values.get("pending_confirmation_task")
                            data["task_name"] = pending_task.get("task_name") if pending_task else None
                except Exception as check_err:
                    logger.error(f"Error checking for pending confirmation: {check_err}")
            
            if message_type == "canvas_confirmation":
                action = data.get("action")  # 'confirm' or 'cancel'
                task_name = data.get("task_name")
                
                logger.info(f"📊 Received canvas {action} for task '{task_name}' on thread {thread_id}")
                
                # Send acknowledgment
                await websocket.send_json({
                    "node": "canvas_confirmation_received",
                    "thread_id": thread_id,
                    "data": {
                        "action": action,
                        "task_name": task_name,
                        "status": "acknowledged"
                    },
                    "timestamp": time.time()
                })
                
                if action == "cancel":
                    # User cancelled - abort the action
                    logger.info(f"❌ User cancelled task '{task_name}'")
                    await websocket.send_json({
                        "node": "__end__",
                        "thread_id": thread_id,
                        "data": {
                            "status": "cancelled",
                            "message": f"Task '{task_name}' was cancelled by user",
                            "final_response": f"Action cancelled. The task '{task_name}' was not executed."
                        },
                        "timestamp": time.time()
                    })
                    continue
                
                # User confirmed - resume execution by treating this as a confirmation prompt
                logger.info(f"✅ User confirmed task '{task_name}' - resuming orchestration")

                try:
                    await websocket.send_json({
                        "node": "canvas_confirmation_processed",
                        "thread_id": thread_id,
                        "data": {
                            "action": action,
                            "task_name": task_name,
                            "status": "resuming_execution",
                            "message": "Applying changes..."
                        },
                        "timestamp": time.time()
                    })
                except Exception as send_err:
                    logger.debug(f"Failed to send canvas_confirmation_processed update: {send_err}")

                # Feed a confirmation prompt into the normal orchestration flow.
                # orchestrator.graph.analyze_request will detect pending_confirmation and route accordingly.
                prompt = "confirm"
                user_response = None
                message_type = None
            
            if not prompt and not user_response:
                await websocket.send_json({
                    "node": "__error__",
                    "error": "Missing 'prompt' field for new conversation or 'user_response' for continuing",
                    "error_type": "ValidationError",
                    "thread_id": thread_id,
                    "timestamp": time.time()
                })
                continue  # Continue waiting for messages

            logger.info(f"Received {'prompt' if prompt else 'user response'} for thread_id {thread_id}")

            # Register UserThread in DB BEFORE sending __start__ so that the
            # 'conversation-created' event that the frontend dispatches on __start__
            # can immediately find this conversation in GET /api/conversations.
            if owner and prompt:
                owner_id_for_reg = owner.get("user_id") or owner.get("sub") if isinstance(owner, dict) else str(owner)
                if owner_id_for_reg:
                    try:
                        from database import SessionLocal as _SL
                        from models import UserThread as _UT
                        _db = _SL()
                        try:
                            existing = _db.query(_UT).filter_by(thread_id=thread_id).first()
                            if not existing:
                                clean_title = (prompt[:97] + "...") if len(prompt) > 100 else prompt
                                _db.add(_UT(thread_id=thread_id, user_id=owner_id_for_reg, title=clean_title))
                                _db.commit()
                                logger.info(f"Pre-registered UserThread {thread_id} for user {owner_id_for_reg}")
                        finally:
                            _db.close()
                    except Exception as _pre_reg_err:
                        logger.warning(f"Pre-registration of UserThread failed (non-fatal): {_pre_reg_err}")

            # Send acknowledgment
            await websocket.send_json({
                "node": "__start__",
                "thread_id": thread_id,
                "message": "Starting agent orchestration..." if prompt else "Continuing conversation...",
                "data": {
                    "original_prompt": prompt,
                    "user_response": user_response,
                    "status": "initializing",
                    "timestamp": time.time()
                }
            })

            sent_todo_snapshot = False

            # Define stream callback for WebSocket updates
            async def stream_callback(node_name: str, node_output, progress: float, node_count: int, thread_id: str):
                try:
                    nonlocal sent_todo_snapshot
                    # Process node data using unified helper
                    final_data = process_node_data(node_name, node_output, progress, node_count, thread_id)

                    # Send enhanced node update using safe sender
                    await safe_websocket_send(websocket, {
                        "node": node_name,
                        "data": final_data,
                        "thread_id": thread_id,
                        "status": "completed",
                        "timestamp": time.time()
                    }, thread_id)
                    logger.info(f"Streamed update from node '{node_name}' (#{node_count}) for thread_id {thread_id} - Progress: {progress:.1f}%")

                    if isinstance(node_output, dict) and node_output.get("todo_list") is not None:
                        todo_list = serialize_complex_object(node_output.get("todo_list", []))
                        todo_statuses = {}
                        for task in todo_list:
                            if isinstance(task, dict):
                                task_key = task.get("id") or task.get("task_id")
                                if task_key:
                                    todo_statuses[task_key] = {
                                        "status": task.get("status"),
                                        "description": task.get("description"),
                                        "updated_at": task.get("updated_at")
                                    }

                        await safe_websocket_send(websocket, {
                            "node": "todo_list_update",
                            "thread_id": thread_id,
                            "data": {
                                "mode": "snapshot" if not sent_todo_snapshot else "delta",
                                "todo_list": todo_list,
                                "task_statuses": todo_statuses,
                                "current_task_id": node_output.get("current_task_id")
                            },
                            "timestamp": time.time()
                        }, thread_id)
                        sent_todo_snapshot = True
                    
                    # Emit brain reasoning when Brain.think() completes
                    if node_name == "omni_brain" and isinstance(node_output, dict):
                        decision = node_output.get("decision")
                        if decision and isinstance(decision, dict):
                            reasoning = decision.get("reasoning")
                            if reasoning:
                                await safe_websocket_send(websocket, {
                                    "node": "brain_thinking",
                                    "thread_id": thread_id,
                                    "data": {
                                        "reasoning": reasoning,
                                        "action_type": decision.get("action_type"),
                                        "iteration": node_output.get("iteration")
                                    },
                                    "timestamp": time.time()
                                }, thread_id)
                                logger.debug(f"🧠 Emitted brain reasoning: {reasoning[:100]}...")
                    
                    # Emit task status events for real-time UI updates
                    if node_name == "execute_batch" and isinstance(node_output, dict):
                        task_events = node_output.get("task_events", [])
                        if task_events:
                            logger.info(f"🔄 Emitting {len(task_events)} task status events")
                            for event in task_events:
                                event_type = event.get("event_type")
                                task_name = event.get("task_name")
                                
                                if event_type == "task_started":
                                    # Generate activity description (clean one-liner for UI)
                                    task_desc = event.get("task_description", "")
                                    activity_desc = task_desc.split('.')[0] if task_desc else task_name
                                    if len(activity_desc) > 80:
                                        activity_desc = activity_desc[:77] + "..."
                                    
                                    await safe_websocket_send(websocket, {
                                        "node": "task_started",
                                        "thread_id": thread_id,
                                        "task_name": task_name,
                                        "task_description": task_desc,
                                        "activity_description": activity_desc,
                                        "agent_name": event.get("agent_name"),
                                        "timestamp": event.get("timestamp", time.time())
                                    }, thread_id)
                                    logger.debug(f"🚀 Emitted task_started for '{task_name}'")
                                    
                                elif event_type == "task_completed":
                                    # Include result summary if available  
                                    result_summary = event.get("result_summary", "Completed successfully")
                                    
                                    await safe_websocket_send(websocket, {
                                        "node": "task_completed",
                                        "thread_id": thread_id,
                                        "task_name": task_name,
                                        "task_description": event.get("task_description"),
                                        "result_summary": result_summary,
                                        "agent_name": event.get("agent_name"),
                                        "execution_time": event.get("execution_time", 0),
                                        "timestamp": event.get("timestamp", time.time())
                                    }, thread_id)
                                    logger.debug(f"✅ Emitted task_completed for '{task_name}'")
                                    
                                elif event_type == "task_failed":
                                    await safe_websocket_send(websocket, {
                                        "node": "task_failed",
                                        "thread_id": thread_id,
                                        "task_name": task_name,
                                        "error": event.get("error"),
                                        "execution_time": event.get("execution_time", 0),
                                        "timestamp": event.get("timestamp", time.time())
                                    }, thread_id)
                except Exception as e:
                    logger.error(f"Error in stream_callback: {e}", exc_info=True)
            
            # Monotonic sequence counter for task events — lets the frontend
            # enforce delivery order (task_started always before task_completed).
            _task_event_seq = {"n": 0}

            def _next_seq() -> int:
                _task_event_seq["n"] += 1
                return _task_event_seq["n"]

            # Define task event callback for REAL-TIME task status streaming
            async def task_event_callback(event: dict):
                """Stream task events in real-time as tasks start/complete"""
                try:
                    event_type = event.get("event_type")
                    task_name = event.get("task_name")
                    seq = _next_seq()

                    if event_type == "task_started":
                        # Generate activity description (clean one-liner for UI)
                        task_desc = event.get("task_description", "")
                        activity_desc = event.get("activity_description") or (
                            task_desc.split('.')[0] if task_desc else task_name
                        )
                        if activity_desc and len(activity_desc) > 80:
                            activity_desc = activity_desc[:77] + "..."

                        await safe_websocket_send(websocket, {
                            "node": "task_started",
                            "seq": seq,
                            "thread_id": thread_id,
                            "task_name": task_name,
                            "task_description": task_desc,
                            "activity_description": activity_desc,
                            "agent_name": event.get("agent_name"),
                            "timestamp": event.get("timestamp", time.time()),
                        }, thread_id)
                        logger.info(f"📡 REAL-TIME: Task started - '{task_name}' (seq={seq})")

                    elif event_type == "task_completed":
                        result_summary = event.get("result_summary", "Completed successfully")

                        await safe_websocket_send(websocket, {
                            "node": "task_completed",
                            "seq": seq,
                            "thread_id": thread_id,
                            "task_name": task_name,
                            "task_description": event.get("task_description"),
                            "result_summary": result_summary,
                            "agent_name": event.get("agent_name"),
                            "execution_time": event.get("execution_time", 0),
                            "timestamp": event.get("timestamp", time.time()),
                        }, thread_id)
                        logger.info(
                            f"📡 REAL-TIME: Task completed - '{task_name}' "
                            f"({event.get('execution_time', 0):.2f}s, seq={seq})"
                        )

                    elif event_type == "task_failed":
                        await safe_websocket_send(websocket, {
                            "node": "task_failed",
                            "seq": seq,
                            "thread_id": thread_id,
                            "task_name": task_name,
                            "task_description": event.get("task_description"),
                            "error": event.get("error"),
                            "agent_name": event.get("agent_name"),
                            "execution_time": event.get("execution_time", 0),
                            "timestamp": event.get("timestamp", time.time()),
                        }, thread_id)
                        logger.warning(
                            f"📡 REAL-TIME: Task failed - '{task_name}': "
                            f"{event.get('error')} (seq={seq})"
                        )

                    elif event_type == "task_progress":
                        # Real-time agent progress message (from SSE stream)
                        message = event.get("message", "")
                        if message:
                            await safe_websocket_send(websocket, {
                                "node": "task_progress",
                                "seq": seq,
                                "thread_id": thread_id,
                                "task_name": task_name,
                                "activity_description": message,
                                "agent_name": event.get("agent_name"),
                                "timestamp": event.get("timestamp", time.time()),
                            }, thread_id)
                            logger.debug(
                                f"📡 REAL-TIME: Task progress - '{task_name}': {message} (seq={seq})"
                            )

                except Exception as e:
                    logger.error(f"Error in task_event_callback: {e}", exc_info=True)

            # Convert files data to FileObject instances with enhanced error handling
            file_objects = []
            if files_data:
                logger.info(f"Processing {len(files_data)} files from WebSocket message")
                for idx, file_data in enumerate(files_data):
                    try:
                        if isinstance(file_data, dict) and 'file_name' in file_data:
                            # Validate required fields
                            required_fields = ['file_name', 'file_path', 'file_type']
                            missing_fields = [f for f in required_fields if f not in file_data]
                            if missing_fields:
                                logger.warning(f"File {idx} missing fields: {missing_fields}. Skipping.")
                                continue
                            
                            file_objects.append(FileObject(
                                file_name=file_data['file_name'],
                                file_path=file_data['file_path'],
                                file_type=file_data['file_type']
                            ))
                            logger.info(f"Added file {idx}: {file_data['file_name']} at {file_data['file_path']}")
                        else:
                            logger.warning(f"File {idx} not a dict or missing 'file_name' field. Type: {type(file_data)}")
                    except Exception as file_err:
                        logger.error(f"Failed to create FileObject from file {idx} ({file_data.get('file_name', 'unknown')}): {file_err}")
                        # Continue processing other files
            
            # Register frontend WebSocket for screenshot relay
            # Browser agent connects via /ws/screenshots/{thread_id} and screenshots are relayed here
            async with frontend_ws_lock:
                frontend_websockets[thread_id] = websocket
                logger.info(f"📸 Frontend WebSocket registered for screenshot relay: thread={thread_id}")
            
            try:
                # Use unified orchestration service with streaming and enhanced error handling
                try:
                    final_state = await execute_orchestration(
                        prompt=prompt,
                        thread_id=thread_id,
                        user_response=user_response,
                        files=file_objects if file_objects else None,
                        stream_callback=stream_callback,
                        task_event_callback=task_event_callback,
                        planning_mode=planning_mode,
                        owner=owner  # Pass owner for user thread registration
                    )
                except asyncio.TimeoutError:
                    logger.error(f"Orchestration execution timed out for thread_id {thread_id}")
                    await websocket.send_json({
                        "node": "__error__",
                        "thread_id": thread_id,
                        "error": "Request processing timed out. Please try with a simpler request.",
                        "error_type": "TimeoutError",
                        "message": "An error occurred during orchestration: Request processing timed out",
                        "status": "error",
                        "timestamp": time.time()
                    })
                    continue
                except Exception as orch_err:
                    logger.error(f"Error executing orchestration for thread_id {thread_id}: {orch_err}", exc_info=True)
                    error_type = type(orch_err).__name__
                    user_friendly_message = "An unexpected error occurred during orchestration"
                    if "permission" in str(orch_err).lower():
                        user_friendly_message = "Permission denied. Please check your authentication."
                    elif "not found" in str(orch_err).lower():
                        user_friendly_message = "Required resource not found."
                    elif "invalid" in str(orch_err).lower():
                        user_friendly_message = "Invalid request parameters."
                    
                    await websocket.send_json({
                        "node": "__error__",
                        "thread_id": thread_id,
                        "error": str(orch_err)[:200],  # Limit error message length
                        "error_type": error_type,
                        "message": f"An error occurred during orchestration: {user_friendly_message}",
                        "status": "error",
                        "timestamp": time.time()
                    })
                    continue
            finally:
                # Cleanup - always cleanup regardless of success/failure
                polling_active = False

            # Check if workflow is paused for user input
            if final_state.get("pending_user_input"):
                # Check for Omni-Dispatcher approval
                pending_approval = final_state.get("pending_approval", False)
                pending_decision = final_state.get("pending_decision")
                
                # Check if this is an approval request
                needs_approval = final_state.get("needs_approval", False) or pending_approval
                
                # Calculate cost if approval is needed
                estimated_cost = 0.0
                task_count = 0
                if needs_approval:
                    task_plan = final_state.get("task_plan", [])
                    for batch in task_plan:
                        for task_dict in batch:
                            task_count += 1
                            if isinstance(task_dict, dict):
                                primary_agent = task_dict.get('primary', {})
                                cost = primary_agent.get('price_per_call_usd', 0.0)
                                if cost:
                                    estimated_cost += cost
                
                # Send appropriate node event for frontend
                node_event = "action_approval_required" if pending_approval else "__user_input_required__"
                
                await websocket.send_json({
                    "node": node_event,
                    "thread_id": thread_id,
                    "data": {
                        "question_for_user": final_state.get("question_for_user"),
                        "approval_required": needs_approval,
                        "pending_approval": pending_approval,
                        "pending_decision": pending_decision,
                        "estimated_cost": estimated_cost,
                        "task_count": task_count,
                        "task_plan": final_state.get("task_plan", []),
                        "task_agent_pairs": final_state.get("task_agent_pairs", []),
                        # Forward Omni-Dispatcher fields
                        "execution_plan": final_state.get("execution_plan"),
                        "action_history": final_state.get("action_history"),
                        "insights": final_state.get("insights")
                    },
                    "message": "Additional information required to complete your request.",
                    "status": "pending_user_input",
                    "timestamp": time.time()
                })
                logger.info(f"WebSocket workflow paused for user input in thread_id {thread_id}, needs_approval: {needs_approval}, pending_approval: {pending_approval}")
                continue  # Continue waiting for user response message

            # NOTE: Conversation history is already saved in execute_orchestration()
            # No need to save again here - avoid redundancy
            
            # Get serializable state with error handling
            try:
                from backend.orchestrator.graph import get_serializable_state
                serializable_state = get_serializable_state(final_state, thread_id)
            except Exception as serialize_err:
                logger.warning(f"Failed to serialize complete state for thread {thread_id}: {serialize_err}")
                # Fallback: send a minimal but valid response
                serializable_state = {
                    "status": "completed",
                    "thread_id": thread_id,
                    "message": "Orchestration completed successfully",
                    "warning": f"Some data could not be processed: {str(serialize_err)[:100]}"
                }

            # Try to send the __end__ event, but handle if connection is already closed
            try:
                # Explicitly surface the most important fields so the frontend
                # doesn't have to dig through the nested state dict.
                end_payload = {
                    "node": "__end__",
                    "thread_id": thread_id,
                    # --- Top-level convenience fields ---
                    "final_response": final_state.get("final_response"),
                    "canvas_display": (
                        final_state.get("canvas_data")
                        or final_state.get("canvas_display")
                    ),
                    "canvas_type": final_state.get("canvas_type"),
                    "canvas_title": final_state.get("canvas_title"),
                    "has_canvas": final_state.get("has_canvas", False),
                    # --- Full serialized state for components that need it ---
                    "data": serializable_state,
                    "message": "Agent orchestration completed successfully.",
                    "status": "completed",
                    "timestamp": time.time(),
                }
                await safe_websocket_send(websocket, end_payload, thread_id)
                logger.info(f"WebSocket stream completed successfully for thread_id {thread_id}")
            except Exception as send_error:
                # Connection might be closed already - that's okay, the data is saved in the database
                logger.warning(f"Could not send __end__ event (connection closed): {send_error}")
                logger.info(f"Orchestration completed for thread_id {thread_id}, but client disconnected before receiving final message")
            
            # Keep the connection open for multi-turn conversations
            # The frontend will continue to send messages or close the connection when done

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for thread_id {thread_id}")
    except Exception as e:
        # Enhanced error handling with categorization and logging
        error_type = type(e).__name__
        error_message = str(e)
        
        # Categorize error for better logging
        error_category = "unknown"
        if "database" in error_message.lower() or "db" in error_message.lower():
            error_category = "database"
        elif "permission" in error_message.lower() or "unauthorized" in error_message.lower():
            error_category = "authorization"
        elif "timeout" in error_message.lower():
            error_category = "timeout"
        elif "resource" in error_message.lower() or "not found" in error_message.lower():
            error_category = "resource_not_found"
        elif "invalid" in error_message.lower():
            error_category = "validation"
        
        logger.error(f"WebSocket error for thread_id {thread_id} [Category: {error_category}]: {error_type} - {error_message}", exc_info=True)
        
        # Prepare user-friendly error message
        user_message = "An error occurred during orchestration"
        if error_category == "database":
            user_message = "Database connection error. Please try again later."
        elif error_category == "authorization":
            user_message = "You do not have permission to perform this action."
        elif error_category == "timeout":
            user_message = "Request took too long. Please try with a simpler request."
        elif error_category == "resource_not_found":
            user_message = "A required resource was not found."
        
        # Truncate error details for security
        error_details = error_message[:150] if len(error_message) > 150 else error_message
        
        # Use safe send to handle closed connections gracefully
        sent = await safe_websocket_send(websocket, {
            "node": "__error__",
            "thread_id": thread_id or "unknown",
            "error": error_details,
            "error_type": error_type,
            "error_category": error_category,
            "message": user_message,
            "status": "error",
            "timestamp": time.time()
        }, thread_id or "unknown")
        
        if sent:
            logger.info(f"Error response sent to client for thread_id {thread_id}")
        else:
            logger.debug(f"Could not send error message (connection closed) for thread_id {thread_id}")

# NOTE: Agent routes (/api/agents/*) are defined in routers/agents_router.py
# and included via app.include_router(agents_router.router)

@app.get("/api/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}

@app.get("/api/metrics/dashboard")
async def get_dashboard_metrics(request: Request, db: Session = Depends(get_db)):
    """Get comprehensive dashboard metrics for the current user"""
    try:
        # Get user ID from request headers
        user_id = request.headers.get("X-User-ID")
        if not user_id:
            raise HTTPException(status_code=401, detail="User ID not provided")
        
        from datetime import datetime, timedelta
        import os
        import json
        
        # Get conversation count
        conversation_count = db.query(UserThread).filter(UserThread.user_id == user_id).count()
        
        # Get workflow count
        workflow_count = db.query(Workflow).filter(Workflow.user_id == user_id).count()
        
        # Get total agents
        agent_count = db.query(Agent).filter(Agent.status == StatusEnum.active).count()
        
        # Time periods
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = now - timedelta(days=7)
        month_start = now - timedelta(days=30)
        yesterday = now - timedelta(days=1)
        
        # Get recent activity (last 24 hours)
        recent_activity = db.query(UserThread).filter(
            UserThread.user_id == user_id,
            UserThread.created_at >= yesterday
        ).count()
        
        # Get conversation trend (last 7 days)
        conversation_trend = []
        for i in range(6, -1, -1):
            date = now - timedelta(days=i)
            date_str = date.strftime('%Y-%m-%d')
            count = db.query(UserThread).filter(
                UserThread.user_id == user_id,
                UserThread.created_at >= date,
                UserThread.created_at < date + timedelta(days=1)
            ).count()
            conversation_trend.append({
                "date": date.strftime('%b %d'),
                "count": count
            })
        
        # Get workflow status distribution
        active_workflows = db.query(Workflow).filter(
            Workflow.user_id == user_id,
            Workflow.status == 'active'
        ).count()
        archived_workflows = db.query(Workflow).filter(
            Workflow.user_id == user_id,
            Workflow.status == 'archived'
        ).count()
        
        workflow_status = []
        if active_workflows > 0:
            workflow_status.append({"name": "Active", "value": active_workflows})
        if archived_workflows > 0:
            workflow_status.append({"name": "Archived", "value": archived_workflows})
        
        # Get recent conversations
        recent_conversations = db.query(UserThread).filter(
            UserThread.user_id == user_id
        ).order_by(UserThread.updated_at.desc()).limit(5).all()
        
        recent_conv_list = [
            {
                "id": conv.thread_id,
                "title": conv.title or "Untitled Conversation",
                "date": conv.created_at.strftime('%Y-%m-%d'),
                "status": "completed"
            }
            for conv in recent_conversations
        ]
        
        # === NEW METRICS ===
        
        # 1. Cost Tracking - Parse conversation history files to calculate costs
        cost_today = 0.0
        cost_week = 0.0
        cost_month = 0.0
        total_cost = 0.0
        total_tasks = 0
        successful_tasks = 0
        failed_tasks = 0
        total_response_time = 0.0
        response_time_count = 0
        agent_usage = {}
        agent_costs = {}
        hourly_usage = [0] * 24
        
        # Get all user conversations
        all_conversations = db.query(UserThread).filter(
            UserThread.user_id == user_id
        ).all()
        
        for conv in all_conversations:
            # Try to load conversation history file
            history_file = os.path.join(CONVERSATION_HISTORY_DIR, f"{conv.thread_id}.json")
            if os.path.exists(history_file):
                try:
                    with open(history_file, 'r') as f:
                        history_data = json.load(f)
                        
                        # Extract task_agent_pairs if available
                        task_pairs = history_data.get('task_agent_pairs', [])
                        
                        for pair in task_pairs:
                            agent_name = pair.get('primary', {}).get('name', 'Unknown')
                            agent_id = pair.get('primary', {}).get('id')
                            
                            # Get agent cost
                            if agent_id:
                                agent = db.query(Agent).filter(Agent.id == agent_id).first()
                                if agent:
                                    cost = agent.price_per_call_usd
                                    total_cost += cost
                                    
                                    # Track agent usage and costs
                                    if agent_name not in agent_usage:
                                        agent_usage[agent_name] = 0
                                        agent_costs[agent_name] = 0.0
                                    agent_usage[agent_name] += 1
                                    agent_costs[agent_name] += cost
                                    
                                    # Time-based cost tracking
                                    if conv.created_at >= today_start:
                                        cost_today += cost
                                    if conv.created_at >= week_start:
                                        cost_week += cost
                                    if conv.created_at >= month_start:
                                        cost_month += cost
                            
                            total_tasks += 1
                        
                        # Track completed tasks (simplified - assume all tasks in history are completed)
                        if history_data.get('final_response'):
                            successful_tasks += len(task_pairs)
                        
                        # Track hourly usage
                        hour = conv.created_at.hour
                        hourly_usage[hour] += 1
                        
                except Exception as e:
                    logger.warning(f"Could not parse history file {history_file}: {e}")
        
        # Calculate success rate
        success_rate = (successful_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        # Calculate average response time (mock for now - would need actual timing data)
        avg_response_time = 2.5  # minutes (placeholder)
        
        # Get top agents by usage
        top_agents = sorted(agent_usage.items(), key=lambda x: x[1], reverse=True)[:5]
        top_agents_list = [
            {
                "name": name,
                "calls": calls,
                "cost": agent_costs.get(name, 0.0),
                "cost_per_call": agent_costs.get(name, 0.0) / calls if calls > 0 else 0
            }
            for name, calls in top_agents
        ]
        
        # Hourly usage pattern
        hourly_pattern = [
            {"hour": f"{i:02d}:00", "count": hourly_usage[i]}
            for i in range(24)
        ]
        
        # Cost trend (last 7 days)
        cost_trend = []
        for i in range(6, -1, -1):
            date = now - timedelta(days=i)
            day_cost = 0.0
            
            day_conversations = db.query(UserThread).filter(
                UserThread.user_id == user_id,
                UserThread.created_at >= date,
                UserThread.created_at < date + timedelta(days=1)
            ).all()
            
            for conv in day_conversations:
                history_file = os.path.join(CONVERSATION_HISTORY_DIR, f"{conv.thread_id}.json")
                if os.path.exists(history_file):
                    try:
                        with open(history_file, 'r') as f:
                            history_data = json.load(f)
                            task_pairs = history_data.get('task_agent_pairs', [])
                            for pair in task_pairs:
                                agent_id = pair.get('primary', {}).get('id')
                                if agent_id:
                                    agent = db.query(Agent).filter(Agent.id == agent_id).first()
                                    if agent:
                                        day_cost += agent.price_per_call_usd
                    except:
                        pass
            
            cost_trend.append({
                "date": date.strftime('%b %d'),
                "cost": round(day_cost, 4)
            })
        
        return {
            # Existing metrics
            "total_conversations": conversation_count,
            "total_workflows": workflow_count,
            "total_agents": agent_count,
            "recent_activity": recent_activity,
            "conversation_trend": conversation_trend,
            "workflow_status": workflow_status,
            "recent_conversations": recent_conv_list,
            
            # New metrics
            "cost_metrics": {
                "today": round(cost_today, 4),
                "week": round(cost_week, 4),
                "month": round(cost_month, 4),
                "total": round(total_cost, 4),
                "avg_per_conversation": round(total_cost / conversation_count, 4) if conversation_count > 0 else 0
            },
            "cost_trend": cost_trend,
            "performance_metrics": {
                "total_tasks": total_tasks,
                "successful_tasks": successful_tasks,
                "failed_tasks": failed_tasks,
                "success_rate": round(success_rate, 2),
                "avg_response_time": avg_response_time,
                "avg_tasks_per_conversation": round(total_tasks / conversation_count, 2) if conversation_count > 0 else 0
            },
            "top_agents": top_agents_list,
            "hourly_usage": hourly_pattern
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching dashboard metrics: {e}")
        logger.exception("Full traceback:")
        raise HTTPException(status_code=500, detail=f"Failed to fetch metrics: {str(e)}")

@app.get("/api/agent-servers/status")
async def get_agent_servers_status():
    """Get the status of all agent servers"""
    async with agent_status_lock:
        status_copy = {
            name: {
                'port': info['port'],
                'status': info['status'],
                'pid': info['process'].pid if info['process'] else None
            }
            for name, info in agent_status.items()
        }
    return status_copy

# Global list to track agent processes and their status
agent_processes = []
agent_status = {}  # {agent_name: {'port': int, 'process': subprocess.Popen, 'status': 'starting'|'ready'|'failed'}}
agent_status_lock = asyncio.Lock()

def cleanup_agents():
    """Stop all agent processes"""
    global agent_processes
    for process in agent_processes:
        try:
            if process.poll() is not None:
                continue

            if platform.system() == "Windows":
                try:
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                        check=False,
                    )
                except Exception:
                    process.terminate()
                    process.wait(timeout=5)
            else:
                process.terminate()
                process.wait(timeout=5)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
    agent_processes = []
    agent_status.clear()

async def wait_for_agent_ready(agent_name: str, port: int, timeout: float = 30.0) -> bool:
    """
    Wait for a specific agent to be ready by checking its health endpoint.
    Returns True if agent is ready, False if timeout or failed.
    """
    import time
    
    start_time = time.time()
    # Try /health first (standard), then fall back to / for legacy agents
    health_endpoints = [f"http://localhost:{port}/health", f"http://localhost:{port}/"]
    
    while time.time() - start_time < timeout:
        async with agent_status_lock:
            status = agent_status.get(agent_name, {}).get('status')
            if status == 'ready':
                return True
            elif status == 'failed':
                return False
        
        # Try to connect to the agent using multiple endpoints
        for health_url in health_endpoints:
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    response = await client.get(health_url)
                    if response.status_code == 200:
                        async with agent_status_lock:
                            if agent_name in agent_status:
                                agent_status[agent_name]['status'] = 'ready'
                        return True
            except:
                pass
        
        await asyncio.sleep(0.5)
    
    async with agent_status_lock:
        if agent_name in agent_status:
            agent_status[agent_name]['status'] = 'failed'
    return False

async def check_agent_health_background():
    """Background task to check agent health and update status"""
    
    while True:
        await asyncio.sleep(3)  # Check every 3 seconds (reduced frequency)
        
        async with agent_status_lock:
            agents_to_check = list(agent_status.items())
        
        for agent_name, info in agents_to_check:
            if info['status'] == 'starting':
                port = info['port']
                # Try /health first (standard), then fall back to / for legacy agents
                health_endpoints = [f"http://localhost:{port}/health", f"http://localhost:{port}/"]
                
                for health_url in health_endpoints:
                    try:
                        async with httpx.AsyncClient(timeout=1.0) as client:
                            response = await client.get(health_url)
                            if response.status_code == 200:
                                async with agent_status_lock:
                                    agent_status[agent_name]['status'] = 'ready'
                                break  # Stop checking other endpoints once ready
                    except:
                        pass

def start_agents_async():
    """Start agents asynchronously without blocking main.py startup"""
    global agent_processes, agent_status
    
    # Clean up any existing agents first
    cleanup_agents()
    
    # Use absolute path based on the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    agents_dir = os.path.join(project_root, "agents")
    
    if not os.path.isdir(agents_dir):
        logger.warning(f"'{agents_dir}' directory not found. Skipping agent server startup.")
        return

    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)

    # Check for SKIP_AGENTS environment variable (comma-separated list)
    skip_agents_env = os.getenv("SKIP_AGENTS", "")
    skip_agents = [a.strip().lower() for a in skip_agents_env.split(",") if a.strip()]
    if skip_agents:
        logger.info(f"SKIP_AGENTS configured: {skip_agents}")

    agent_files = [f for f in os.listdir(agents_dir) if f.endswith("_agent.py")]
    logger.info(f"Starting {len(agent_files)} agent server(s)...")

    for agent_file in agent_files:
        agent_path = os.path.join(agents_dir, agent_file)
        agent_name = agent_file.replace('.py', '')
        port = None
        
        # Check if this agent should be skipped (for running in separate terminal)
        if agent_name.lower() in skip_agents or agent_file.lower() in skip_agents:
            logger.info(f"Skipping {agent_name} (in SKIP_AGENTS, run it separately)")
            continue
        
        try:
            # Extract port from agent file
            with open(agent_path, 'r', encoding='utf-8') as f:
                content = f.read()
                match = re.search(r'port\s*=\s*int\(os\.getenv\([^,]+,\s*(\d+)\)', content)
                if not match:
                    match = re.search(r"port\s*=\s*(\d+)", content)
                if match:
                    port = int(match.group(1))

            if port is None:
                continue

            log_path = os.path.join(logs_dir, f"{agent_file}.log")

            # Start the agent process
            with open(log_path, 'w') as log_file:
                if platform.system() == "Windows":
                    process = subprocess.Popen(
                        [sys.executable, agent_path],
                        stdout=log_file,
                        stderr=log_file,
                        creationflags=subprocess.CREATE_NO_WINDOW
                    )
                else:
                    process = subprocess.Popen(
                        [sys.executable, agent_path],
                        stdout=log_file,
                        stderr=log_file,
                        start_new_session=True
                    )
                
                agent_processes.append(process)
                agent_status[agent_name] = {
                    'port': port,
                    'process': process,
                    'status': 'starting'
                }
        
        except Exception:
            agent_status[agent_name] = {
                'port': port,
                'process': None,
                'status': 'failed'
            }
    
    # Register cleanup handler
    import atexit
    atexit.register(cleanup_agents)
    
    logger.info("Agent servers started. Ready for requests.")

# Note: Startup logic has been moved to the lifespan context manager above.
# The @app.on_event("startup") decorator is deprecated in favor of lifespan.

if __name__ == "__main__":
    # Agents will be started automatically via the lifespan context manager
    # Run the main FastAPI app
    import uvicorn
    # Use 0.0.0.0 to bind to all interfaces (fixes IPv4/IPv6 issues)
    # Add ws_ping_interval and ws_ping_timeout for better WebSocket stability
    reload_enabled = os.getenv("UVICORN_RELOAD", "0").strip().lower() in {"1", "true", "yes", "on"}
    if reload_enabled:
        logger.info("Starting backend with auto-reload enabled (UVICORN_RELOAD=1)")
    else:
        logger.info("Starting backend with auto-reload disabled (set UVICORN_RELOAD=1 to enable)")

    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",  # Changed from 127.0.0.1 to support both IPv4 and IPv6
            port=8000,
            reload=reload_enabled,
            reload_includes=["*.py", "orchestrator/*.py", "agents/*.py", "services/*.py", "routers/*.py"],  # Watch all Python files
            ws_ping_interval=20,  # Send ping every 20 seconds
            ws_ping_timeout=20,   # Wait 20 seconds for pong
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down backend...")
    finally:
        cleanup_agents()
