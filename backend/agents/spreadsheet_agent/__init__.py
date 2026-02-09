"""
Spreadsheet Agent v3.0

Unified spreadsheet operations with LLM-powered task decomposition.
Only 4 endpoints: /execute, /continue, /health, /files
"""

import os
import sys
from pathlib import Path

# ==================== ROBUST PATH HANDLING ====================
# Allow running this script directly or as a package
PACKAGE_DIR = Path(__file__).parent.absolute()
AGENTS_DIR = PACKAGE_DIR.parent
BACKEND_DIR = AGENTS_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

# Ensure correct paths in sys.path
# Order: PROJECT_ROOT > BACKEND_DIR > AGENTS_DIR > PACKAGE_DIR
for path in [str(PROJECT_ROOT), str(BACKEND_DIR), str(AGENTS_DIR), str(PACKAGE_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)


import logging
from typing import Optional, Dict, Any, Union
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import AGENT_PORT, AGENT_VERSION, logger
from .schemas import (
    ExecuteRequest, ContinueRequest, ExecuteResponse,
    HealthResponse, TaskStatus
)
from .agent import spreadsheet_agent
from .state import session_state

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Spreadsheet Agent",
    version=AGENT_VERSION,
    description="Unified spreadsheet operations with LLM-powered task decomposition"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint."""
    return HealthResponse()


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check with stats."""
    stats = session_state.get_stats()
    return HealthResponse(cache_stats=stats)


@app.post("/execute", response_model=ExecuteResponse)
async def execute(
    # Form fields for flexibility
    prompt: Optional[str] = Form(None),
    query: Optional[str] = Form(None),        # Support 'query' field from orchestrator
    instruction: Optional[str] = Form(None),  # Support 'instruction' field
    action: Optional[str] = Form(None),
    params: Optional[str] = Form(None),  # JSON string
    thread_id: Optional[str] = Form(None),
    task_id: Optional[str] = Form(None),
    file_id: Optional[str] = Form(None), # Support file_id from orchestrator
    # File upload or path
    file: Union[UploadFile, str, None] = File(None)
):
    """
    Unified execution endpoint.
    
    Handles ALL spreadsheet operations:
    - File upload: Include 'file' parameter (UploadFile or path string)
    - Natural language: Include 'prompt' parameter  
    - Direct action: Include 'action' and 'params' parameters
    
    Examples:
    - Upload: POST with file
    - Query: POST with prompt="What is total revenue by category?"
    - Transform: POST with prompt="Add a profit margin column"
    - Direct: POST with action="filter", params='{"column": "Status", "value": "Active"}'
    """
    try:
        # Parse params if provided as JSON string
        params_dict = {}
        if params:
            import json
            try:
                params_dict = json.loads(params)
            except:
                params_dict = {"raw": params}
        
        # Add explicit form fields to params_dict so _extract_prompt can find them
        if query:
            params_dict['query'] = query
        if instruction:
            params_dict['instruction'] = instruction
        if file_id:
            params_dict['file_id'] = file_id
        
        # Handle file upload or path
        file_content = None
        filename = None
        
        # Normalize file input: use file_id as fallback for file path if file is None
        file_to_process = file
        if not file_to_process and file_id:
             file_to_process = file_id
        
        if file_to_process:
            if isinstance(file_to_process, UploadFile):
                file_content = await file_to_process.read()
                filename = file_to_process.filename
            elif isinstance(file_to_process, str):
                # Handle local file path sent as string
                import os
                
                # normalize slashes
                file_path = file_to_process.replace('\\', '/')
                filename = os.path.basename(file_path)
                
                # Paths to check
                paths_to_check = [
                    file_path,  # Direct path
                    os.path.join(r"d:\Internship\Orbimesh\storage\spreadsheets", filename),
                    os.path.join(r"d:\Internship\Orbimesh\storage\spreadsheet_agent", filename),
                    os.path.join(r"d:\Internship\Orbimesh\storage", filename),
                    os.path.abspath(filename) # Check current working dir
                ]
                
                found_path = None
                for path in paths_to_check:
                    if os.path.exists(path):
                        found_path = path
                        break
                
                if found_path:
                    try:
                        with open(found_path, 'rb') as f:
                            file_content = f.read()
                        logger.info(f"Loaded local file from path: {found_path}")
                    except Exception as e:
                        logger.error(f"Failed to read local file {found_path}: {e}")
                else:
                    # Silence warning if it looks like a UUID (which is expected for file_ids)
                    import re
                    is_uuid = bool(re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', str(file_to_process)))
                    if not is_uuid:
                        logger.warning(f"File path provided but not found in any common locations: {file_to_process}")
                        logger.warning(f"Checked: {paths_to_check}")
        
        # Execute
        return await spreadsheet_agent.execute(
            prompt=prompt,
            action=action,
            params=params_dict,
            thread_id=thread_id or "default",
            task_id=task_id,
            file_content=file_content,
            filename=filename
        )
        
    except Exception as e:
        logger.error(f"Execute endpoint error: {e}")
        return ExecuteResponse(
            status=TaskStatus.ERROR,
            success=False,
            error=str(e)
        )


@app.post("/execute/json", response_model=ExecuteResponse)
async def execute_json(request: ExecuteRequest):
    """
    JSON body version of /execute for programmatic access.
    """
    try:
        # Inject file_path/file_id into params if provided
        params = request.params or {}
        if request.file_path:
            params['file_path'] = request.file_path
        if request.file_id:
            params['file_id'] = request.file_id

        # Merge extra fields from flattened request into params
        # This captures params like 'column_name', 'instruction' sent by orchestrator at top level
        known_fields = {'prompt', 'action', 'params', 'thread_id', 'task_id', 'file_content', 'filename', 'file_path', 'file_id'}
        
        # Access extra fields if enabled in config
        request_data = request.dict()
        for key, value in request_data.items():
            if key not in known_fields and value is not None:
                params[key] = value
                
        # Preserve instruction if it came in keys
        if 'instruction' in params and not params.get('instruction'): 
             pass # preserve what came in
            
        return await spreadsheet_agent.execute(
            prompt=request.prompt,
            action=request.action,
            params=params,
            thread_id=request.thread_id or "default",
            task_id=request.task_id,
            file_content=request.file_content,
            filename=request.filename
        )
    except Exception as e:
        logger.error(f"Execute JSON endpoint error: {e}")
        return ExecuteResponse(
            status=TaskStatus.ERROR,
            success=False,
            error=str(e)
        )


@app.post("/continue", response_model=ExecuteResponse)
async def continue_task(request: ContinueRequest):
    """
    Resume a paused task with user input.
    
    When /execute returns status='needs_input', call this endpoint
    with the task_id and user's response.
    """
    try:
        return await spreadsheet_agent.continue_task(
            task_id=request.task_id,
            user_response=request.user_response,
            thread_id=request.thread_id or "default"
        )
    except Exception as e:
        logger.error(f"Continue endpoint error: {e}")
        return ExecuteResponse(
            status=TaskStatus.ERROR,
            success=False,
            error=str(e)
        )


@app.get("/files")
async def list_files(thread_id: Optional[str] = None):
    """
    List files in session.
    """
    try:
        thread_id = thread_id or "default"
        session = session_state.get(thread_id)
        
        if not session:
            return {"files": [], "count": 0}
        
        files = []
        for file_id, metadata in session.file_metadata.items():
            files.append({
                "file_id": file_id,
                "file_path": session.file_paths.get(file_id, ""),
                **metadata
            })
        
        return {"files": files, "count": len(files)}
        
    except Exception as e:
        logger.error(f"List files error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/files/{file_id}")
async def get_file(file_id: str, thread_id: Optional[str] = None):
    """
    Get file metadata.
    """
    try:
        thread_id = thread_id or "default"
        session = session_state.get(thread_id)
        
        if not session or file_id not in session.file_metadata:
            raise HTTPException(status_code=404, detail="File not found")
        
        return {
            "file_id": file_id,
            "file_path": session.file_paths.get(file_id, ""),
            **session.file_metadata[file_id]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get file error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# STARTUP / SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup():
    """Initialize on startup."""
    logger.info(f"Spreadsheet Agent v{AGENT_VERSION} starting on port {AGENT_PORT}")
    logger.info("Endpoints: /execute, /continue, /health, /files")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    logger.info("Spreadsheet Agent shutting down")
    # Cleanup expired sessions
    removed = session_state.cleanup_expired()
    if removed:
        logger.info(f"Cleaned up {removed} expired sessions")


# ============================================================================
# FOR RUNNING DIRECTLY
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=AGENT_PORT)
