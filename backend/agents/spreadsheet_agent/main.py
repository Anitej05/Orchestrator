"""
Main FastAPI application for the Spreadsheet Agent.

This module consolidates all API routes and uses the modular components.
"""

import os
import sys
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional
from asyncio import Lock as AsyncLock
import anyio

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Header, Request
from fastapi.responses import FileResponse
from dotenv import load_dotenv

# Add parent directory to path for imports
CURRENT_DIR = Path(__file__).parent
BACKEND_DIR = CURRENT_DIR.parent.parent
sys.path.insert(0, str(BACKEND_DIR))

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import modular components - using absolute imports
from agents.spreadsheet_agent.config import STORAGE_DIR, AGENT_PORT, MAX_FILE_SIZE_MB
from agents.spreadsheet_agent.models import (
    ApiResponse,
    CreateSpreadsheetRequest,
    NaturalLanguageQueryRequest,
    CompareFilesRequest,
    MergeFilesRequest,
    ComparisonResult
)

# Import contract models from backend
from models import StandardResponse, StandardResponseMetrics, DecisionContract

from agents.spreadsheet_agent.memory import spreadsheet_memory
from agents.spreadsheet_agent import session as session_module  # Import module for _thread_local access
from agents.spreadsheet_agent.session import (
    get_conversation_dataframes,
    get_conversation_file_paths,
    ensure_file_loaded,
    get_dataframe_state,
    store_dataframe,
    get_dataframe
)
from agents.spreadsheet_agent.llm_agent import query_agent
from agents.spreadsheet_agent.code_generator import generate_modification_code, generate_csv_from_instruction
from agents.spreadsheet_agent.display import dataframe_to_canvas, format_dataframe_preview
from agents.spreadsheet_agent.utils import (
    validate_file,
    load_dataframe,
    dataframe_to_csv,
    dataframe_to_excel,
    serialize_dataframe,
    convert_numpy_types,
    handle_execution_error
)

# Import metrics display using absolute import from backend
# Backend path is already added to sys.path above (line 22)
try:
    # Try standard import first
    from utils.metrics_display import display_execution_metrics, display_session_metrics
except ImportError:
    # If that fails, use importlib to load directly
    import importlib.util
    metrics_path = BACKEND_DIR / "utils" / "metrics_display.py"
    spec = importlib.util.spec_from_file_location("metrics_display", metrics_path)
    metrics_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics_module)
    display_execution_metrics = metrics_module.display_execution_metrics
    display_session_metrics = metrics_module.display_session_metrics

# Import standardized file manager
try:
    from agents.utils.agent_file_manager import AgentFileManager, FileType, FileStatus
except ImportError:
    try:
        from utils.agent_file_manager import AgentFileManager, FileType, FileStatus
    except ImportError:
        logger.error("Failed to import agent_file_manager")
        raise

# Import session manager (now in same directory)
from agents.spreadsheet_agent.spreadsheet_session_manager import spreadsheet_session_manager

# Import main agent class for orchestrator endpoints
from agents.spreadsheet_agent.agent import spreadsheet_agent

# Create FastAPI app
app = FastAPI(title="Spreadsheet Agent", version="2.0.0")

# Create storage directory
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# Metrics tracking
import time


# ============================================================================
# DECISION CONTRACT VALIDATION (Phase 3: Refactor)
# ============================================================================

def validate_decision_contract(
    contract: Optional[Dict[str, Any]],
    instruction: str,
    endpoint: str
) -> Optional[Dict[str, Any]]:
    """
    Validate Decision Contract against request.
    Returns error dict if validation fails, None if valid.
    
    The orchestrator is the SOLE authority on task classification.
    This function enforces orchestrator decisions.
    """
    if not contract:
        # No contract provided - allow (backward compatibility)
        return None
    
    # Check write permissions
    if not contract.get('allow_write', False):
        write_keywords = ['add', 'remove', 'delete', 'drop', 'modify', 'change', 'update', 'insert', 'create']
        if any(kw in instruction.lower() for kw in write_keywords):
            return {
                "success": False,
                "needs_clarification": True,
                "message": f"Contract forbids write operations. Instruction: {instruction[:100]}"
            }
    
    # Check schema change permissions
    if not contract.get('allow_schema_change', False):
        schema_keywords = ['rename column', 'drop column', 'add column', 'merge', 'join']
        if any(kw in instruction.lower() for kw in schema_keywords):
            return {
                "success": False,
                "needs_clarification": True,
                "message": f"Contract forbids schema changes. Instruction: {instruction[:100]}"
            }
    
    return None
from collections import defaultdict

call_metrics = {
    "api_calls": defaultdict(int),
    "api_timing": defaultdict(list),
    "api_errors": defaultdict(int),
    "llm_calls": {
        "total": 0,
        "by_provider": defaultdict(int)
    },
    "start_time": time.time()
}

# Initialize file manager
file_manager = AgentFileManager(
    agent_id="spreadsheet_agent",
    storage_dir=str(STORAGE_DIR),
    default_ttl_hours=None,
    auto_cleanup=True,
    cleanup_interval_hours=24
)

# Lock for pandas operations
spreadsheet_operation_lock = AsyncLock()

# Legacy fallback storage
dataframes: Dict[str, pd.DataFrame] = {}
file_paths: Dict[str, str] = {}


# --- Middleware for Metrics Tracking ---

@app.middleware("http")
async def track_requests(request, call_next):
    """Track all HTTP requests for metrics."""
    start_time = time.time()
    endpoint = request.url.path
    
    try:
        response = await call_next(request)
        
        # Track successful call
        call_metrics["api_calls"][endpoint] += 1
        duration = time.time() - start_time
        call_metrics["api_timing"][endpoint].append(duration)
        
        # Keep only last 100 timings per endpoint to avoid memory growth
        if len(call_metrics["api_timing"][endpoint]) > 100:
            call_metrics["api_timing"][endpoint] = call_metrics["api_timing"][endpoint][-100:]
        
        return response
    except Exception as e:
        # Track error
        call_metrics["api_errors"][endpoint] += 1
        duration = time.time() - start_time
        call_metrics["api_timing"][endpoint].append(duration)
        raise


# --- Startup and Shutdown Events ---

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    logger.info("🚀 Spreadsheet Agent v2.0 starting up...")
    logger.info(f"📁 Storage directory: {STORAGE_DIR}")
    logger.info(f"🔌 Agent port: {AGENT_PORT}")
    
    # Load memory cache from disk
    try:
        spreadsheet_memory.load_from_disk()
        logger.info("✅ Memory cache loaded")
    except Exception as e:
        logger.warning(f"⚠️ Could not load memory cache: {e}")
    
    # Log LLM provider status
    if query_agent.providers:
        provider_names = ' → '.join([p['name'] for p in query_agent.providers])
        logger.info(f"🤖 LLM providers initialized: {provider_names}")
    else:
        logger.warning("⚠️ No LLM providers available")
    
    logger.info("✅ Spreadsheet Agent ready")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    logger.info("👋 Shutting down Spreadsheet Agent...")
    
    # Save memory cache to disk
    try:
        spreadsheet_memory.save_to_disk()
        logger.info("✅ Memory cache saved")
    except Exception as e:
        logger.error(f"❌ Failed to save memory cache: {e}")
    
    logger.info("✅ Shutdown complete")


# --- API Endpoints ---

@app.post("/upload", response_model=StandardResponse)
async def upload_file(
    file: UploadFile = File(...),
    thread_id: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    orchestrator_content_id: Optional[str] = Form(None)
):
    """Upload a CSV or Excel file"""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Invalid file: filename is required")

        # Validate file
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in [".csv", ".xlsx", ".xls"]:
            raise HTTPException(
                status_code=400, 
                detail="Invalid file type. Please upload a .csv, .xlsx, or .xls file."
            )

        # Read file content
        file_content = await file.read()
        
        # Validate file size
        if len(file_content) > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE_MB}MB."
            )
        
        # Register file
        metadata = await file_manager.register_file(
            content=file_content,
            filename=file.filename,
            file_type=FileType.SPREADSHEET,
            thread_id=thread_id,
            user_id=user_id,
            orchestrator_content_id=orchestrator_content_id,
            custom_metadata={
                "original_filename": file.filename,
                "content_type": file.content_type
            },
            tags=["spreadsheet", "uploaded"]
        )
        
        file_id = metadata.file_id
        file_location = metadata.storage_path

        # Load into pandas
        df = load_dataframe(file_location)

        # Store in session
        conversation_thread_id = thread_id or "default"
        store_dataframe(file_id, df, file_location, conversation_thread_id)
        
        # Legacy fallback
        dataframes[file_id] = df
        file_paths[file_id] = file_location

        # Mark as processed
        file_manager.mark_as_processed(
            file_id=file_id,
            processing_result={
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "dtypes": {k: str(v) for k, v in df.dtypes.to_dict().items()}
            }
        )

        logger.info(f"File '{file.filename}' uploaded with file_id: {file_id}")

        # Generate canvas display
        canvas_display = dataframe_to_canvas(
            df=df,
            title=f"Uploaded: {file.filename}",
            filename=file.filename,
            display_mode='full',
            max_rows=50,
            file_id=file_id,
            metadata={'operation': 'upload'}
        )

        return StandardResponse(
            success=True,
            route="/upload",
            task_type="create",
            data={
                "file_id": file_id,
                "filename": file.filename,
                "file_path": file_location,
                "rows": len(df),
                "columns": len(df.columns),
                "orchestrator_format": metadata.to_orchestrator_format()
            },
            preview={"canvas_display": canvas_display},
            artifact=None,
            metrics=StandardResponseMetrics(
                rows_processed=len(df),
                columns_affected=len(df.columns)
            ),
            confidence=1.0,
            message=f"File '{file.filename}' uploaded successfully with {len(df)} rows and {len(df.columns)} columns"
        )

    except Exception as e:
        logger.error(f"File upload failed: {e}", exc_info=True)
        return StandardResponse(
            success=False,
            route="/upload",
            task_type="create",
            data={},
            message=f"Upload failed: {str(e)}"
        )


@app.post("/nl_query", response_model=StandardResponse)
async def natural_language_query(
    request: NaturalLanguageQueryRequest,
    thread_id: Optional[str] = Header(None, alias="x-thread-id")
):
    """Process natural language query against spreadsheet data"""
    try:
        file_id = request.file_id
        question = request.question
        max_iterations = request.max_iterations
        
        thread_id = thread_id or "default"
        
        if not file_id:
            logger.error("❌ [NL_QUERY] Missing file_id in request")
            raise HTTPException(status_code=400, detail="file_id is required. Upload a spreadsheet and provide its file_id.")

        logger.info(f"🚀 [NL_QUERY] Starting with: file_id={file_id}, question='{question[:80]}...'")
        
        # GUARD: Block summary/preview/schema requests (these go to /get_summary or /display)
        question_lower = question.lower()
        redirect_keywords = [
            'summarize', 'summary', 'preview', 'display', 'show me',
            'schema', 'columns', 'describe file', 'what is this', 'what is in',
            'list columns', 'file structure', 'overview'
        ]
        
        if any(kw in question_lower for kw in redirect_keywords):
            logger.warning(f"❌ [NL_QUERY GUARD] Blocked summary/preview request: {question[:100]}")
            return StandardResponse(
                success=False,
                route="/nl_query",
                task_type="qa",
                data={},
                needs_clarification=True,
                message=(
                    f"This endpoint is for analytical questions (why/how/anomaly detection) only. "
                    f"For summaries, previews, or schema information, use /get_summary or /display. "
                    f"Your question: {question[:100]}"
                )
            )
        
        # Ensure file is loaded
        if not ensure_file_loaded(file_id, thread_id, file_manager):
            logger.error(f"❌ [NL_QUERY] File not found: {file_id}")
            raise HTTPException(status_code=404, detail=f"File {file_id} not found")
        
        # Get dataframe
        df = get_dataframe(file_id, thread_id)
        if df is None:
            logger.error(f"❌ [NL_QUERY] Failed to load dataframe: {file_id}")
            raise HTTPException(status_code=500, detail="Failed to load dataframe")
        
        logger.info(f"📊 [NL_QUERY] Dataframe loaded: {len(df)} rows × {len(df.columns)} cols")
        
        # Get session context
        session_context = ""
        if thread_id != "default":
            try:
                session_history = spreadsheet_session_manager.get_session_history(
                    file_id=file_id,
                    thread_id=thread_id,
                    limit=5
                )
                if session_history:
                    session_context = "\n".join([
                        f"{i+1}. {op.get('operation', 'unknown')}: {op.get('description', '')}"
                        for i, op in enumerate(session_history)
                    ])
                    logger.info(f"📜 [NL_QUERY] Found {len(session_history)} previous operations in context")
            except Exception as e:
                logger.warning(f"⚠️  [NL_QUERY] Could not get session history: {e}")

        # Execute query
        async with spreadsheet_operation_lock:
            logger.info(f"🤖 [NL_QUERY] Sending to LLM for processing (max_iterations={max_iterations})...")
            
            result = await query_agent.query(
                df=df,
                question=question,
                max_iterations=max_iterations,
                session_context=session_context,
                file_id=file_id,
                thread_id=thread_id
            )
            
            if result.success:
                logger.info(f"✅ [NL_QUERY] LLM processing completed successfully")
            else:
                logger.error(f"❌ [NL_QUERY] LLM processing failed: {result.error}")
        
        # Read-only: do NOT persist dataframe mutations from /nl_query
        
        # Track operation
        if thread_id != "default":
            try:
                spreadsheet_session_manager.track_operation(
                    thread_id=thread_id,
                    file_id=file_id,
                    operation="nl_query",
                    description=f"Query: {question[:100]}",
                    result_summary={"success": result.success, "answer": result.answer[:200]}
                )
            except Exception as e:
                logger.warning(f"Could not track operation: {e}")
        
        # Generate canvas display if we have final data
        canvas_display = None
        if result.final_data and isinstance(result.final_data, list):
            try:
                result_df = pd.DataFrame(result.final_data)
                canvas_display = dataframe_to_canvas(
                    df=result_df,
                    title=f"Query Result: {question[:50]}",
                    filename=f"query_result.csv",
                    display_mode='full',
                    max_rows=50,
                    metadata={'operation': 'nl_query', 'question': question}
                )
            except Exception as e:
                logger.warning(f"Could not generate canvas: {e}")
        
        response_data = {
            "question": result.question,
            "answer": result.answer,
            "steps_taken": result.steps_taken,
            "final_data": result.final_data,
            "success": result.success,
            "error": result.error
        }
        
        if canvas_display:
            response_data["canvas_display"] = canvas_display
        
        # Display execution metrics for debugging
        if hasattr(result, 'metrics') and result.metrics:
            display_execution_metrics(
                metrics=result.metrics,
                agent_name="Spreadsheet",
                operation_name="nl_query",
                success=result.success
            )
        
        logger.info(f"📤 [NL_QUERY] Returning response to orchestrator")
        return StandardResponse(
            success=bool(result.success),
            route="/nl_query",
            task_type="qa",
            data=response_data,
            preview={"canvas_display": canvas_display} if canvas_display else None,
            artifact=None,
            metrics=StandardResponseMetrics(
                llm_calls=result.metrics.get('llm_calls', 0) if hasattr(result, 'metrics') and result.metrics else 0
            ),
            confidence=result.confidence if hasattr(result, 'confidence') else 0.8,
            message=result.answer if result.success else (result.error or "Query failed")
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Natural language query failed: {e}", exc_info=True)
        return StandardResponse(
            success=False,
            route="/nl_query",
            task_type="qa",
            data={},
            message=f"Query processing failed: {str(e)}"
        )


@app.post("/transform", response_model=ApiResponse)
async def transform_data(
    file_id: str = Form(...),
    operation: Optional[str] = Form(None),
    params: Optional[str] = Form(None),
    instruction: Optional[str] = Form(None),
    thread_id: Optional[str] = Form(None)
):
    """Transform spreadsheet data using operation/params or natural language instruction"""
    try:
        thread_id = thread_id or "default"
        
        # Ensure file is loaded
        if not ensure_file_loaded(file_id, thread_id, file_manager):
            raise HTTPException(status_code=404, detail=f"File {file_id} not found")
        
        # Get dataframe
        df = get_dataframe(file_id, thread_id)
        if df is None:
            raise HTTPException(status_code=500, detail="Failed to load dataframe")
        
        # Handle different input formats
        if operation and params:
            # Use operation/params format (as per agent entry)
            try:
                import json
                params_dict = json.loads(params)
                
                if operation == "group":
                    # Handle groupby operation
                    group_col = params_dict.get("group_by", "Product Category")
                    agg_col = params_dict.get("aggregate", "Quantity")
                    agg_func = params_dict.get("function", "sum")
                    
                    if group_col in df.columns and agg_col in df.columns:
                        if agg_func == "sum":
                            result_df = df.groupby(group_col)[agg_col].sum().reset_index()
                        elif agg_func == "mean":
                            result_df = df.groupby(group_col)[agg_col].mean().reset_index()
                        elif agg_func == "count":
                            result_df = df.groupby(group_col)[agg_col].count().reset_index()
                        else:
                            result_df = df.groupby(group_col)[agg_col].sum().reset_index()
                        
                        modified_df = result_df
                        code = f"df.groupby('{group_col}')['{agg_col}'].{agg_func}().reset_index()"
                    else:
                        raise ValueError(f"Columns not found: {group_col}, {agg_col}")
                else:
                    raise ValueError(f"Operation '{operation}' not implemented")
                    
            except Exception as e:
                return ApiResponse(success=False, error=f"Failed to execute operation: {str(e)}")
        
        elif instruction:
            # Use natural language instruction
            code = await generate_modification_code(df, instruction)
            if not code:
                return ApiResponse(success=False, error="Failed to generate transformation code")
            
            # Execute code
            async with spreadsheet_operation_lock:
                try:
                    local_vars = {"df": df, "pd": pd}
                    exec(code, {"__builtins__": {}}, local_vars)
                    modified_df = local_vars.get("df", df)
                except Exception as e:
                    error_msg = handle_execution_error(e, code)
                    return ApiResponse(success=False, error=error_msg)
        
        else:
            # No operation, params, or instruction provided - use default aggregation
            logger.warning("No operation/params or instruction provided to /transform, using default category aggregation")
            if "Product Category" in df.columns and "Quantity" in df.columns:
                modified_df = df.groupby("Product Category")["Quantity"].sum().reset_index()
                code = "df.groupby('Product Category')['Quantity'].sum().reset_index()"
            else:
                return ApiResponse(success=False, error="No operation specified and default columns not found")
        
        # Update dataframe
        thread_paths = get_conversation_file_paths(thread_id)
        store_dataframe(file_id, modified_df, thread_paths.get(file_id, file_paths.get(file_id, "")), thread_id)
        
        # Track operation
        if thread_id != "default":
            try:
                operation_desc = instruction or f"{operation} with {params}"
                spreadsheet_session_manager.track_operation(
                    thread_id=thread_id,
                    file_id=file_id,
                    operation="transform",
                    description=operation_desc,
                    result_summary={"rows": len(modified_df), "columns": len(modified_df.columns)}
                )
            except Exception as e:
                logger.warning(f"Could not track operation: {e}")
        
        # Generate canvas display
        metadata = file_manager.get_file(file_id)
        filename = metadata.original_name if metadata else "transformed.csv"
        
        canvas_display = dataframe_to_canvas(
            df=modified_df,
            title=f"Transformed: {filename}",
            filename=filename,
            display_mode='full',
            max_rows=50,
            file_id=file_id,
            metadata={'operation': 'transform', 'instruction': instruction or f"{operation}({params})"}
        )
        
        return ApiResponse(success=True, result={
            "file_id": file_id,
            "rows": len(modified_df),
            "columns": len(modified_df.columns),
            "code_executed": code if 'code' in locals() else f"operation: {operation}",
            "canvas_display": canvas_display
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transform failed: {e}", exc_info=True)
        return ApiResponse(success=False, error=str(e))


@app.post("/get_summary", response_model=StandardResponse)
async def get_summary(file_id: str = Form(...), show_preview: bool = Form(False), thread_id: Optional[str] = Form(None)):
    """Get summary of spreadsheet with headers, dtypes, and preview"""
    try:
        thread_id = thread_id or "default"
        
        if not ensure_file_loaded(file_id, thread_id, file_manager):
            raise HTTPException(status_code=404, detail="File not found")
        
        df = get_dataframe(file_id, thread_id)
        if df is None:
            raise HTTPException(status_code=500, detail="Failed to load dataframe")
        
        metadata = file_manager.get_file(file_id)
        filename = metadata.original_name if metadata else "unknown.csv"
        
        summary = {
            "filename": filename,
            "headers": df.columns.tolist(),
            "rows": df.head(5).to_dict(orient="records"),
            "dtypes": {k: str(v) for k, v in df.dtypes.to_dict().items()},
            "total_rows": len(df),
            "total_columns": len(df.columns)
        }
        
        canvas_display = None
        if show_preview:
            canvas_display = dataframe_to_canvas(
                df=df,
                title=f"Preview: {filename}",
                filename=filename,
                display_mode='full',
                max_rows=10,
                file_id=file_id
            )
        
        return StandardResponse(
            success=True,
            route="/get_summary",
            task_type="summary",
            data=summary,
            preview={"canvas_display": canvas_display} if canvas_display else None,
            artifact=None,
            metrics=StandardResponseMetrics(
                rows_processed=len(df),
                columns_affected=len(df.columns)
            ),
            confidence=1.0,
            message=f"Summary retrieved for {filename}: {len(df)} rows, {len(df.columns)} columns"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get summary failed: {e}", exc_info=True)
        return StandardResponse(
            success=False,
            route="/get_summary",
            task_type="summary",
            data={},
            message=f"Failed to get summary: {str(e)}"
        )


@app.post("/get_summary_with_canvas", response_model=ApiResponse)
async def get_summary_with_canvas(file_id: str = Form(...), max_rows: int = Form(10), thread_id: Optional[str] = Form(None)):
    """Get summary with automatic canvas display"""
    try:
        # Call get_summary with show_preview=True
        return await get_summary(file_id=file_id, show_preview=True, thread_id=thread_id)
    
    except Exception as e:
        logger.error(f"Get summary with canvas failed: {e}", exc_info=True)
        return ApiResponse(success=False, error=str(e))


@app.post("/query", response_model=ApiResponse)
async def query_data(file_id: str = Form(...), query: str = Form(...), thread_id: Optional[str] = Form(None)):
    """Execute pandas query on spreadsheet"""
    try:
        thread_id = thread_id or "default"
        
        if not ensure_file_loaded(file_id, thread_id, file_manager):
            raise HTTPException(status_code=404, detail="File not found")
        
        df = get_dataframe(file_id, thread_id)
        if df is None:
            raise HTTPException(status_code=500, detail="Failed to load dataframe")
        
        # Execute query
        async with spreadsheet_operation_lock:
            result_df = df.query(query)
        
        return ApiResponse(success=True, result={
            "query": query,
            "result": result_df.head(100).to_dict(orient="records"),
            "rows_returned": len(result_df)
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        return ApiResponse(success=False, error=str(e))


@app.post("/get_column_stats", response_model=ApiResponse)
async def get_column_stats(file_id: str = Form(...), column_name: str = Form(...), thread_id: Optional[str] = Form(None)):
    """Get descriptive statistics for a specific column"""
    try:
        thread_id = thread_id or "default"
        
        if not ensure_file_loaded(file_id, thread_id, file_manager):
            raise HTTPException(status_code=404, detail="File not found")
        
        df = get_dataframe(file_id, thread_id)
        if df is None:
            raise HTTPException(status_code=500, detail="Failed to load dataframe")
        
        if column_name not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{column_name}' not found")
        
        stats = df[column_name].describe().to_dict()
        
        return ApiResponse(success=True, result={"column": column_name, "stats": stats})
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get column stats failed: {e}", exc_info=True)
        return ApiResponse(success=False, error=str(e))


@app.post("/display", response_model=ApiResponse)
async def display_spreadsheet(
    file_id: str = Form(...),
    display_mode: str = Form('full'),
    max_rows: int = Form(100),
    thread_id: Optional[str] = Form(None)
):
    """Display spreadsheet in canvas format"""
    try:
        thread_id = thread_id or "default"
        
        if not ensure_file_loaded(file_id, thread_id, file_manager):
            raise HTTPException(status_code=404, detail="File not found")
        
        df = get_dataframe(file_id, thread_id)
        if df is None:
            raise HTTPException(status_code=500, detail="Failed to load dataframe")
        
        metadata = file_manager.get_file(file_id)
        filename = metadata.original_name if metadata else "unknown.csv"
        
        # Apply display mode
        if display_mode == 'head':
            display_df = df.head(max_rows)
            title = f"{filename} (First {max_rows} rows)"
        elif display_mode == 'tail':
            display_df = df.tail(max_rows)
            title = f"{filename} (Last {max_rows} rows)"
        elif display_mode == 'sample':
            display_df = df.sample(min(max_rows, len(df)))
            title = f"{filename} (Random {max_rows} rows)"
        else:
            display_df = df.head(max_rows) if len(df) > max_rows else df
            title = filename
        
        canvas_display = dataframe_to_canvas(
            df=display_df,
            title=title,
            filename=filename,
            display_mode=display_mode,
            max_rows=max_rows,
            file_id=file_id
        )
        
        return StandardResponse(
            success=True,
            route="/display",
            task_type="preview",
            data={
                "file_id": file_id,
                "filename": filename,
                "rows_displayed": len(display_df)
            },
            preview={"canvas_display": canvas_display},
            artifact=None,
            metrics=StandardResponseMetrics(
                rows_processed=len(display_df),
                columns_affected=len(df.columns)
            ),
            confidence=1.0,
            message=f"Displaying {len(display_df)} rows in {display_mode} mode"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Display failed: {e}", exc_info=True)
        return StandardResponse(
            success=False,
            route="/display",
            task_type="preview",
            data={},
            message=f"Display failed: {str(e)}"
        )


@app.get("/download/{file_id}")
async def download_spreadsheet(file_id: str, format: str = 'xlsx', thread_id: Optional[str] = Header(None, alias="x-thread-id")):
    """Download spreadsheet in specified format (xlsx, csv, json)"""
    try:
        from fastapi.responses import StreamingResponse
        import io
        
        thread_id = thread_id or "default"
        
        if not ensure_file_loaded(file_id, thread_id, file_manager):
            raise HTTPException(status_code=404, detail="File not found")
        
        df = get_dataframe(file_id, thread_id)
        if df is None:
            raise HTTPException(status_code=500, detail="Failed to load dataframe")
        
        metadata = file_manager.get_file(file_id)
        filename = Path(metadata.original_name).stem if metadata else "download"
        
        if format == 'csv':
            output = io.StringIO()
            df.to_csv(output, index=False)
            output.seek(0)
            return StreamingResponse(
                iter([output.getvalue()]),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={filename}.csv"}
            )
        elif format == 'json':
            output = io.StringIO()
            df.to_json(output, orient='records', indent=2)
            output.seek(0)
            return StreamingResponse(
                iter([output.getvalue()]),
                media_type="application/json",
                headers={"Content-Disposition": f"attachment; filename={filename}.json"}
            )
        else:  # xlsx
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Sheet1')
            output.seek(0)
            return StreamingResponse(
                iter([output.getvalue()]),
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": f"attachment; filename={filename}.xlsx"}
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/execute_pandas", response_model=ApiResponse)
async def execute_pandas(
    file_id: str = Form(...),
    instruction: Optional[str] = Form(None),
    pandas_code: Optional[str] = Form(None),
    thread_id: Optional[str] = Form(None)
):
    """Execute pandas code or generate from instruction"""
    try:
        thread_id = thread_id or "default"
        
        if not ensure_file_loaded(file_id, thread_id, file_manager):
            raise HTTPException(status_code=404, detail="File not found")
        
        df = get_dataframe(file_id, thread_id)
        if df is None:
            raise HTTPException(status_code=500, detail="Failed to load dataframe")
        
        # Handle missing instruction - provide a helpful default for aggregation tasks
        if not instruction and not pandas_code:
            # If no instruction provided, try to infer from common aggregation patterns
            logger.warning("No instruction provided, attempting default category aggregation")
            instruction = "List all categories in the 'Product Category' column and sum the 'Quantity' for each category"
        
        # Generate code if instruction provided
        if instruction and not pandas_code:
            pandas_code = await generate_modification_code(df, instruction)
            if not pandas_code:
                return ApiResponse(success=False, error="Failed to generate code")
        
        if not pandas_code:
            raise HTTPException(status_code=400, detail="Either instruction or pandas_code required")
        
        # Execute code
        async with spreadsheet_operation_lock:
            try:
                local_vars = {"df": df, "pd": pd}
                exec(pandas_code, {"__builtins__": {}}, local_vars)
                modified_df = local_vars.get("df", df)
            except Exception as e:
                error_msg = handle_execution_error(e, pandas_code)
                return ApiResponse(success=False, error=error_msg)
        
        # Update dataframe
        thread_paths = get_conversation_file_paths(thread_id)
        store_dataframe(file_id, modified_df, thread_paths.get(file_id, file_paths.get(file_id, "")), thread_id)
        
        return ApiResponse(success=True, result={
            "message": "Code executed successfully",
            "code": pandas_code,
            "rows": len(modified_df),
            "columns": len(modified_df.columns)
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Execute pandas failed: {e}", exc_info=True)
        return ApiResponse(success=False, error=str(e))


@app.post("/simulate_operation", response_model=ApiResponse)
async def simulate_operation_endpoint(
    file_id: str = Form(...),
    pandas_code: str = Form(...),
    thread_id: Optional[str] = Form(None)
):
    """
    Simulate a pandas operation without modifying actual data.
    Returns preview of changes, warnings, and observation data.
    """
    try:
        thread_id = thread_id or "default"
        
        if not ensure_file_loaded(file_id, thread_id, file_manager):
            raise HTTPException(status_code=404, detail="File not found")
        
        df = get_dataframe(file_id, thread_id)
        if df is None:
            raise HTTPException(status_code=500, detail="Failed to load dataframe")
        
        # Import simulation module
        from agents.spreadsheet_agent.simulate import preview_operation
        
        # Run simulation
        result = preview_operation(df, pandas_code, max_preview_rows=20)
        
        return ApiResponse(
            success=result["success"],
            result={
                "simulation": result,
                "message": "Simulation completed. Review changes before applying." if result["success"] else "Simulation failed."
            },
            error=result.get("error")
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Simulation failed: {e}", exc_info=True)
        return ApiResponse(success=False, error=str(e))


@app.post("/plan_operation")
async def plan_operation_endpoint(
    file_id: str = Form(...),
    instruction: Optional[str] = Form(None),
    thread_id: Optional[str] = Form(None),
    stage: str = Form("propose")  # propose, revise, simulate, execute
):
    """
    Multi-stage planning endpoint:
    - propose: Generate initial plan from instruction
    - revise: Revise plan based on feedback
    - simulate: Test plan on copy of data
    - execute: Apply plan to actual data
    
    FAST PATH: For analysis-only requests (summary/describe/analyze), returns immediate response without multi-stage workflow.
    """
    try:
        thread_id = thread_id or "default"
        
        if not ensure_file_loaded(file_id, thread_id, file_manager):
            raise HTTPException(status_code=404, detail="File not found")
        
        df = get_dataframe(file_id, thread_id)
        if df is None:
            raise HTTPException(status_code=500, detail="Failed to load dataframe")
        
        # Handle missing instruction - provide a helpful default for aggregation tasks
        if not instruction:
            logger.warning("No instruction provided to /plan_operation, attempting default category aggregation")
            instruction = "List all categories in the 'Product Category' column and sum the 'Quantity' for each category"
        
        # === MULTI-STAGE PATH: All operations require planning workflow ===
        logger.info(f"📋 [PLAN_OPERATION] Multi-stage path: stage={stage}")
        
        # Import planner
        from agents.spreadsheet_agent.planner import planner
        
        # Stage: PROPOSE
        if stage == "propose":
            # Generate DataFrame context
            df_context = {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "sample": df.head(5).to_dict(orient='records')
            }
            
            # Propose plan
            plan = await planner.propose_plan(df, instruction, df_context)
            
            # Format actions as table for display
            canvas_data = {
                "headers": ["Step", "Action", "Description"],
                "rows": [
                    [
                        str(i + 1),
                        action.action_type.replace("_", " ").title(),
                        getattr(action, 'description', f"Execute {action.action_type}")
                    ]
                    for i, action in enumerate(plan.actions)
                ]
            }
            
            # Return with canvas_display for orchestrator approval
            return ApiResponse(
                success=True,
                result={
                    "status": "plan_ready",
                    "plan_id": plan.plan_id,
                    "message": f"Generated plan with {len(plan.actions)} actions. Review and approve to execute."
                },
                canvas_display={
                    "canvas_type": "spreadsheet_plan",
                    "canvas_title": "Spreadsheet Execution Plan",
                    "canvas_data": canvas_data,
                    "plan_summary": plan.reasoning,
                    "estimated_steps": len(plan.actions),
                    "requires_confirmation": True,
                    "confirmation_message": "Review the plan and approve to proceed with execution"
                }
            )
        
        # Stage: REVISE
        elif stage == "revise":
            # Get plan_id from instruction (should be formatted as JSON)
            try:
                import json
                logger.info(f"[REVISE] Received instruction: {repr(instruction)}")
                revision_data = json.loads(instruction)
                plan_id = revision_data.get("plan_id")
                feedback = revision_data.get("feedback", "")
                logger.info(f"[REVISE] Parsed plan_id: {plan_id}, feedback: {feedback}")
            except Exception as e:
                logger.error(f"[REVISE] Failed to parse instruction: {e}, instruction={repr(instruction)}")
                raise HTTPException(status_code=400, detail=f"For 'revise' stage, instruction must be JSON with plan_id and feedback. Error: {str(e)}")
            
            # Find plan in history
            plan = planner.history.get_plan(plan_id)
            if not plan:
                raise HTTPException(status_code=404, detail=f"Plan {plan_id} not found")
            
            # Revise plan
            revised_plan = await planner.revise_plan(plan, feedback, df)
            
            return ApiResponse(
                success=True,
                result={
                    "plan": revised_plan.to_dict(),
                    "message": f"Revised plan (revision {len(revised_plan.revisions)}). Review and proceed to 'simulate' stage."
                }
            )
        
        # Stage: SIMULATE
        elif stage == "simulate":
            # Get plan_id from instruction
            try:
                import json
                logger.info(f"[SIMULATE] Received instruction: {repr(instruction)}")
                sim_data = json.loads(instruction)
                plan_id = sim_data.get("plan_id")
                logger.info(f"[SIMULATE] Parsed plan_id: {plan_id}")
            except Exception as e:
                logger.error(f"[SIMULATE] Failed to parse instruction: {e}, instruction={repr(instruction)}")
                raise HTTPException(status_code=400, detail=f"For 'simulate' stage, instruction must be JSON with plan_id. Error: {str(e)}")
            
            # Find plan
            plan = planner.history.get_plan(plan_id)
            if not plan:
                raise HTTPException(status_code=404, detail=f"Plan {plan_id} not found")
            
            # Simulate plan
            sim_result = planner.simulate_plan(plan, df)
            
            return ApiResponse(
                success=sim_result["success"],
                result={
                    "plan_id": plan_id,
                    "simulation": sim_result,
                    "message": "Simulation complete. Review warnings and proceed to 'execute' stage if acceptable." if sim_result["success"] else "Simulation failed. Revise plan."
                },
                error=sim_result.get("error")
            )
        
        # Stage: EXECUTE
        elif stage == "execute":
            # Get plan_id and force flag from instruction
            try:
                import json
                logger.info(f"[EXECUTE] Received instruction: {repr(instruction)}")
                exec_data = json.loads(instruction)
                plan_id = exec_data.get("plan_id")
                force = exec_data.get("force", False)
                logger.info(f"[EXECUTE] Parsed plan_id: {plan_id}, force: {force}")
            except Exception as e:
                logger.error(f"[EXECUTE] Failed to parse instruction: {e}, instruction={repr(instruction)}")
                raise HTTPException(status_code=400, detail=f"For 'execute' stage, instruction must be JSON with plan_id and optional force flag. Error: {str(e)}")
            
            # Find plan
            plan = planner.history.get_plan(plan_id)
            if not plan:
                raise HTTPException(status_code=404, detail=f"Plan {plan_id} not found")
            
            # Execute plan
            modified_df, exec_result = planner.execute_plan(plan, df, force=force)
            
            if exec_result["success"]:
                # Update dataframe in session
                thread_paths = get_conversation_file_paths(thread_id)
                store_dataframe(file_id, modified_df, thread_paths.get(file_id, file_paths.get(file_id, "")), thread_id)
                
                return ApiResponse(
                    success=True,
                    result={
                        "plan_id": plan_id,
                        "status": "executed",
                        "execution": exec_result,
                        "shape": modified_df.shape,
                        "columns": modified_df.columns.tolist(),
                        "preview": modified_df.head(20).to_dict(orient='records'),
                        "message": f"Plan executed successfully. DataFrame shape: {modified_df.shape}"
                    }
                )
            else:
                return ApiResponse(
                    success=False,
                    error=exec_result.get("error"),
                    result={"execution": exec_result}
                )
        
        else:
            raise HTTPException(status_code=400, detail=f"Invalid stage: {stage}. Must be 'propose', 'revise', 'simulate', or 'execute'")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Plan operation failed: {e}", exc_info=True)
        return ApiResponse(success=False, error=str(e))


@app.post("/create", response_model=StandardResponse)
async def create_spreadsheet(request: CreateSpreadsheetRequest):
    """Create a new spreadsheet from content or instruction"""
    try:
        thread_id = request.thread_id or "default"
        
        # **ENDPOINT VALIDATION**: Detect if instruction is actually asking for analysis/summary
        if request.instruction:
            instruction_lower = request.instruction.lower()
            analysis_keywords = ['summarize', 'summary', 'analyze', 'analysis', 'describe', 'explain',
                               'review', 'insights', 'examine', 'interpret', 'report about', 'report on']
            is_analysis = any(keyword in instruction_lower for keyword in analysis_keywords)
            
            # Check if asking about existing file
            references_existing = any(phrase in instruction_lower for phrase in 
                                     ['of the', 'about the', 'from the', '.xlsx', '.csv', 'spreadsheet', 'existing', 'uploaded'])
            
            if is_analysis and references_existing:
                error_msg = (f"❌ /create endpoint called with analysis request. "
                           f"This endpoint creates NEW spreadsheets. "
                           f"For analyzing/summarizing existing files, use /nl_query endpoint instead. "
                           f"Instruction received: {request.instruction[:100]}...")
                logger.error(error_msg)
                return StandardResponse(
                    success=False,
                    route="/create",
                    task_type="create",
                    data={},
                    needs_clarification=True,
                    message="Wrong endpoint: Use /nl_query for analysis/summary tasks, not /create"
                )
        
        # Generate CSV content if needed
        if request.instruction:
            csv_content = await generate_csv_from_instruction(
                request.instruction,
                request.content
            )
            if not csv_content:
                return StandardResponse(
                    success=False,
                    route="/create",
                    task_type="create",
                    data={},
                    message="Failed to generate CSV from instruction"
                )
        elif request.content:
            csv_content = request.content
        else:
            raise HTTPException(status_code=400, detail="Either content or instruction required")
        
        # Parse CSV
        import io
        df = pd.read_csv(io.StringIO(csv_content))
        
        # Generate filename
        filename = request.output_filename or "generated_spreadsheet.csv"
        if not filename.endswith(('.csv', '.xlsx')):
            filename += '.csv'
        
        # Save file
        file_path = STORAGE_DIR / filename
        if request.output_format == 'xlsx':
            df.to_excel(file_path, index=False)
        else:
            df.to_csv(file_path, index=False)
        
        # Register with file manager
        with open(file_path, 'rb') as f:
            file_content = f.read()
        
        metadata = await file_manager.register_file(
            content=file_content,
            filename=filename,
            file_type=FileType.SPREADSHEET,
            thread_id=thread_id,
            tags=["created", "generated"]
        )
        
        file_id = metadata.file_id
        
        # Store in session
        store_dataframe(file_id, df, str(file_path), thread_id)
        dataframes[file_id] = df
        file_paths[file_id] = str(file_path)
        
        # Generate canvas
        canvas_display = dataframe_to_canvas(
            df=df,
            title=f"Created: {filename}",
            filename=filename,
            display_mode='full',
            max_rows=50,
            file_id=file_id
        )
        
        return StandardResponse(
            success=True,
            route="/create",
            task_type="create",
            data={
                "file_id": file_id,
                "filename": filename,
                "file_path": str(file_path),
                "rows": len(df),
                "columns": len(df.columns)
            },
            preview={"canvas_display": canvas_display},
            artifact={
                "id": file_id,
                "filename": filename,
                "url": f"/download/{file_id}"
            },
            metrics=StandardResponseMetrics(
                rows_processed=len(df),
                columns_affected=len(df.columns)
            ),
            confidence=1.0,
            message=f"Created spreadsheet '{filename}' with {len(df)} rows and {len(df.columns)} columns"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Create spreadsheet failed: {e}", exc_info=True)
        return StandardResponse(
            success=False,
            route="/create",
            task_type="create",
            data={},
            message=f"Create failed: {str(e)}"
        )


# File management endpoints

@app.get("/files", response_model=ApiResponse)
async def list_files(status: Optional[str] = None, thread_id: Optional[str] = None):
    """List all files managed by this agent"""
    try:
        from agents.utils.agent_file_manager import FileStatus
        file_status = FileStatus(status) if status else FileStatus.ACTIVE
        files = file_manager.list_files(status=file_status, thread_id=thread_id)
        
        return ApiResponse(success=True, result={
            "files": [f.to_orchestrator_format() for f in files],
            "count": len(files)
        })
    except Exception as e:
        logger.error(f"List files failed: {e}", exc_info=True)
        return ApiResponse(success=False, error=str(e))


@app.get("/files/{file_id}", response_model=ApiResponse)
async def get_file_info(file_id: str):
    """Get detailed information about a specific file"""
    try:
        metadata = file_manager.get_file(file_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="File not found")
        
        result = metadata.to_orchestrator_format()
        if metadata.processing_result:
            result["processing_result"] = metadata.processing_result
        
        return ApiResponse(success=True, result=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get file info failed: {e}", exc_info=True)
        return ApiResponse(success=False, error=str(e))


@app.delete("/files/{file_id}", response_model=ApiResponse)
async def delete_file_endpoint(file_id: str):
    """Delete a file from storage"""
    try:
        # Remove from all thread storages
        if hasattr(session_module, '_thread_local') and hasattr(session_module._thread_local, 'dataframes_by_thread'):
            for thread_dfs in session_module._thread_local.dataframes_by_thread.values():
                thread_dfs.pop(file_id, None)
        
        # Remove from legacy storage
        dataframes.pop(file_id, None)
        file_paths.pop(file_id, None)
        
        # Delete from file manager
        success = file_manager.delete_file(file_id)
        if not success:
            raise HTTPException(status_code=404, detail="File not found")
        
        return ApiResponse(success=True, result={"message": f"File {file_id} deleted"})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete file failed: {e}", exc_info=True)
        return ApiResponse(success=False, error=str(e))


@app.post("/files/{file_id}/reload", response_model=ApiResponse)
async def reload_file_endpoint(file_id: str, thread_id: Optional[str] = Form(None)):
    """Reload a file into memory"""
    try:
        thread_id = thread_id or "default"
        
        metadata = file_manager.get_file(file_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="File not found")
        
        df = load_dataframe(metadata.storage_path)
        store_dataframe(file_id, df, metadata.storage_path, thread_id)
        
        # Also store in legacy
        dataframes[file_id] = df
        file_paths[file_id] = metadata.storage_path
        
        return ApiResponse(success=True, result={
            "message": "File reloaded",
            "file_id": file_id,
            "rows": len(df),
            "columns": len(df.columns)
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reload file failed: {e}", exc_info=True)
        return ApiResponse(success=False, error=str(e))


@app.post("/cleanup", response_model=ApiResponse)
async def cleanup_files(max_age_hours: int = 24):
    """Clean up old/expired files"""
    try:
        from agents.utils.agent_file_manager import FileStatus
        
        expired_count = file_manager.cleanup_expired()
        old_count = file_manager.cleanup_old(max_age_hours=max_age_hours)
        
        # Clean up in-memory dataframes
        active_file_ids = {f.file_id for f in file_manager.list_files(status=FileStatus.ACTIVE)}
        removed_from_memory = 0
        for file_id in list(dataframes.keys()):
            if file_id not in active_file_ids:
                dataframes.pop(file_id, None)
                file_paths.pop(file_id, None)
                removed_from_memory += 1
        
        return ApiResponse(success=True, result={
            "expired_removed": expired_count,
            "old_removed": old_count,
            "memory_cleaned": removed_from_memory
        })
    except Exception as e:
        logger.error(f"Cleanup failed: {e}", exc_info=True)
        return ApiResponse(success=False, error=str(e))


@app.post("/compare", response_model=StandardResponse)
async def compare_files(request: CompareFilesRequest):
    """
    Compare multiple spreadsheet files.
    
    Supports schema comparison, key-based row diff, and full value diff.
    Returns comparison results and optionally creates a diff report artifact.
    
    Uses threadpool for CPU-intensive pandas operations to avoid blocking event loop.
    """
    try:
        from agents.spreadsheet_agent.models import ComparisonResult
        from agents.spreadsheet_agent.multi_file_ops import (
            compare_schemas,
            compare_by_keys,
            detect_key_columns,
            generate_diff_report
        )
        
        logger.info(f"Comparing {len(request.file_ids)} files: {request.file_ids}")
        
        thread_id = request.thread_id or "default"

        # Load all dataframes
        dataframes_dict = {}
        for file_id in request.file_ids:
            df = get_dataframe(file_id, thread_id)
            if df is None:
                return StandardResponse(
                    success=False,
                    route="/compare",
                    task_type="compare",
                    data={},
                    message=f"File {file_id} not found or not loaded"
                )
            dataframes_dict[file_id] = df
        
        # THREADPOOL: Offload CPU-intensive comparison to worker thread
        def _perform_comparison():
            """CPU-intensive comparison work (runs in threadpool)"""
            # Schema comparison (always performed)
            schema_diff = compare_schemas(dataframes_dict)
            
            # Row comparison (if mode requires it)
            row_diff = None
            if request.comparison_mode in ["schema_and_key", "full_diff"]:
                # Determine key columns
                key_cols = request.key_columns
                if not key_cols:
                    # Auto-detect keys from first file
                    first_df = list(dataframes_dict.values())[0]
                    key_cols = detect_key_columns(first_df)
                    logger.info(f"Auto-detected key columns: {key_cols}")
                
                if key_cols:
                    try:
                        row_diff = compare_by_keys(dataframes_dict, key_cols, request.comparison_mode)
                    except Exception as e:
                        logger.warning(f"Row comparison failed: {e}")
                        row_diff = {"error": str(e), "summary": f"Could not perform row comparison: {e}"}
                else:
                    row_diff = {"error": "No key columns found", "summary": "Schema comparison only (no unique keys detected)"}
            
            return schema_diff, row_diff
        
        # Run in threadpool to avoid blocking event loop
        schema_diff, row_diff = await anyio.to_thread.run_sync(_perform_comparison)
        
        # Generate diff report artifact only when explicitly requested via a non-JSON output.
        # Default output_format is 'json' which should return structured JSON in the API response.
        diff_artifact_id = None
        if request.output_format in ["csv", "markdown"]:
            report_content = generate_diff_report(schema_diff, row_diff, request.output_format)

            ext = "csv" if request.output_format == "csv" else "md"
            artifact_filename = f"diff_report_{int(time.time())}.{ext}"
            report_bytes = report_content.encode("utf-8")

            from agents.utils.agent_file_manager import FileType
            artifact = await file_manager.register_file(
                content=report_bytes,
                filename=artifact_filename,
                file_type=FileType.SPREADSHEET if ext == "csv" else FileType.DOCUMENT,
                thread_id=thread_id,
                tags=["diff", "comparison", "spreadsheet"]
            )
            diff_artifact_id = artifact.file_id
            logger.info(f"Created diff report artifact: {diff_artifact_id}")
        
        # Build result
        comparison_result = ComparisonResult(
            file_ids=request.file_ids,
            schema_diff=schema_diff,
            row_diff=row_diff,
            summary=f"{schema_diff.get('summary', '')}\n{row_diff.get('summary', '') if row_diff else ''}",
            diff_artifact_id=diff_artifact_id
        )
        
        # Canvas display for orchestrator integration
        canvas_display = {
            "canvas_type": "json",
            "canvas_data": comparison_result.model_dump(),
            "canvas_title": f"Comparison: {len(request.file_ids)} files",
            "requires_confirmation": False
        }
        
        # Prepare artifact info
        artifact_info = None
        if diff_artifact_id:
            artifact_metadata = file_manager.get_file(diff_artifact_id)
            if artifact_metadata:
                artifact_info = {
                    "id": diff_artifact_id,
                    "filename": artifact_metadata.original_name,
                    "url": f"/download/{diff_artifact_id}"
                }
        
        return StandardResponse(
            success=True,
            route="/compare",
            task_type="compare",
            data=comparison_result.model_dump(),
            preview={"canvas_display": canvas_display},
            artifact=artifact_info,
            metrics=StandardResponseMetrics(
                rows_processed=sum(len(df) for df in dataframes_dict.values()),
                columns_affected=0
            ),
            confidence=1.0,
            message=f"Compared {len(request.file_ids)} files successfully"
        )
    
    except Exception as e:
        logger.error(f"Compare files failed: {e}", exc_info=True)
        return StandardResponse(
            success=False,
            route="/compare",
            task_type="compare",
            data={},
            message=f"Comparison failed: {str(e)}"
        )


@app.post("/merge", response_model=StandardResponse)
async def merge_files(request: MergeFilesRequest):
    """
    Merge multiple spreadsheet files.
    
    Supports join, union, and concatenation.
    Creates a new merged file artifact.
    
    Uses threadpool for CPU-intensive pandas merge operations to avoid blocking event loop.
    """
    try:
        from agents.spreadsheet_agent.multi_file_ops import merge_dataframes
        
        logger.info(f"Merging {len(request.file_ids)} files: {request.file_ids} (mode: {request.merge_type})")
        
        thread_id = request.thread_id or "default"

        # Load all dataframes
        dataframes_dict = {}
        for file_id in request.file_ids:
            df = get_dataframe(file_id, thread_id)
            if df is None:
                return StandardResponse(
                    success=False,
                    route="/merge",
                    task_type="merge",
                    data={},
                    message=f"File {file_id} not found or not loaded"
                )
            dataframes_dict[file_id] = df
        
        # THREADPOOL: Offload CPU-intensive merge to worker thread
        def _perform_merge():
            """CPU-intensive merge work (runs in threadpool)"""
            return merge_dataframes(
                dataframes_dict,
                merge_type=request.merge_type,
                join_type=request.join_type,
                key_columns=request.key_columns
            )
        
        # Run in threadpool to avoid blocking event loop
        merged_df, summary = await anyio.to_thread.run_sync(_perform_merge)
        
        # Save merged file as new artifact
        output_filename = request.output_filename or f"merged_{int(time.time())}.csv"
        if not output_filename.endswith(('.csv', '.xlsx')):
            output_filename += '.csv'
        
        # Build file bytes in threadpool, then register via AgentFileManager (it writes to storage).
        def _build_output_bytes() -> bytes:
            if output_filename.endswith('.csv'):
                return merged_df.to_csv(index=False).encode('utf-8')
            else:
                from io import BytesIO
                buffer = BytesIO()
                merged_df.to_excel(buffer, index=False)
                return buffer.getvalue()

        output_bytes = await anyio.to_thread.run_sync(_build_output_bytes)

        from agents.utils.agent_file_manager import FileType
        artifact = await file_manager.register_file(
            content=output_bytes,
            filename=output_filename,
            file_type=FileType.SPREADSHEET,
            thread_id=thread_id,
            tags=["merged", "spreadsheet"]
        )

        # Store in session (thread-scoped)
        store_dataframe(artifact.file_id, merged_df, artifact.storage_path, thread_id)
        
        logger.info(f"Created merged artifact: {artifact.file_id} ({summary})")
        
        # Canvas display
        canvas_display = {
            "canvas_type": "spreadsheet",
            "canvas_data": {
                "file_id": artifact.file_id,
                "filename": output_filename,
                "shape": merged_df.shape,
                "preview": merged_df.head(10).to_dict(orient="records"),
                "summary": summary
            },
            "canvas_title": f"Merged: {output_filename}",
            "requires_confirmation": False
        }
        
        return StandardResponse(
            success=True,
            route="/merge",
            task_type="merge",
            data={
                "file_id": artifact.file_id,
                "filename": output_filename,
                "shape": merged_df.shape,
                "summary": summary
            },
            preview={"canvas_display": canvas_display},
            artifact={
                "id": artifact.file_id,
                "filename": output_filename,
                "url": f"/download/{artifact.file_id}"
            },
            metrics=StandardResponseMetrics(
                rows_processed=len(merged_df),
                columns_affected=len(merged_df.columns)
            ),
            confidence=1.0,
            message=f"Merged {len(request.file_ids)} files: {summary}"
        )
    
    except Exception as e:
        logger.error(f"Merge files failed: {e}", exc_info=True)
        return StandardResponse(
            success=False,
            route="/merge",
            task_type="merge",
            data={},
            message=f"Merge failed: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "agent": "spreadsheet_agent",
        "llm_providers": len(query_agent.providers),
        "cache_stats": spreadsheet_memory.get_cache_stats()
    }


# ============================================================================
# ORCHESTRATOR-COMPATIBLE ENDPOINTS (Task 10.1)
# ============================================================================

@app.post("/execute")
async def execute_action(
    request: Request,
    message: Optional[Dict[str, Any]] = None,
    instruction: Optional[str] = Form(None),
    file_id: Optional[str] = Form(None),
    thread_id: Optional[str] = Form(None),
    decision_contract: Optional[str] = Form(None)
):
    """
    Unified execution endpoint supporting orchestrator communication.
    
    This endpoint handles requests from the orchestrator and returns AgentResponse format.
    Supports both JSON and form-encoded requests.
    """
    try:
        # Handle both JSON and form-data requests
        content_type = request.headers.get("content-type", "")
        
        if "application/json" in content_type:
            # JSON request - get the body
            body = await request.json()
            message = body
        else:
            # Form data request
            if message is None:
                message = {}
            
            # If we have form parameters, build the message structure
            if instruction or file_id or thread_id:
                if 'payload' not in message:
                    message['payload'] = {}
                if instruction:
                    message['payload']['instruction'] = instruction
                if file_id:
                    message['payload']['file_id'] = file_id
                if thread_id:
                    message['payload']['thread_id'] = thread_id
        
        # Extract payload and parameters
        payload = message.get('payload', {})
        action = message.get('action')
        prompt = payload.get('prompt') or payload.get('instruction')
        
        # Generate task_id if not provided
        task_id = payload.get('task_id', f"task-{int(time.time())}")
        
        logger.info(f"🚀 [EXECUTE] Action={action}, Prompt={prompt[:100] if prompt else 'None'}..., TaskID={task_id}")
        
        # Validate required fields
        file_id_param = payload.get('file_id')
        if not file_id_param:
            logger.error("❌ [EXECUTE] Missing file_id in payload")
            return {
                "status": "error",
                "error": "Missing required field: file_id",
                "context": {"task_id": task_id}
            }
        
        if not prompt:
            logger.error("❌ [EXECUTE] Missing instruction/prompt in payload")
            return {
                "status": "error", 
                "error": "Missing required field: instruction or prompt",
                "context": {"task_id": task_id}
            }
        
        thread_id_param = payload.get('thread_id', 'default')
        
        logger.info(f"📁 [EXECUTE] Processing: file_id={file_id_param}, thread_id={thread_id_param}")
        
        # Ensure file is loaded
        if not ensure_file_loaded(file_id_param, thread_id_param, file_manager):
            logger.error(f"❌ [EXECUTE] File not found: {file_id_param}")
            return {
                "status": "error",
                "error": f"File {file_id_param} not found or could not be loaded",
                "context": {"task_id": task_id, "file_id": file_id_param}
            }
        
        # Get dataframe
        df = get_dataframe(file_id_param, thread_id_param)
        if df is None:
            logger.error(f"❌ [EXECUTE] Failed to load dataframe: {file_id_param}")
            return {
                "status": "error",
                "error": f"Failed to load dataframe for file {file_id_param}",
                "context": {"task_id": task_id, "file_id": file_id_param}
            }
        
        logger.info(f"📊 [EXECUTE] Dataframe loaded: {len(df)} rows × {len(df.columns)} cols")
        
        # Process the instruction
        instruction_lower = prompt.lower()
        
        # Check for analytical/aggregation questions
        analytical_keywords = [
            'list', 'show', 'what', 'how many', 'count', 'total', 'sum', 'average', 'mean',
            'categories', 'category', 'group', 'aggregate', 'analyze', 'analysis',
            'find', 'identify', 'calculate', 'compute', 'determine', 'quantities', 'quantity',
            'scan', 'unique', 'present'
        ]
        
        is_analytical = any(keyword in instruction_lower for keyword in analytical_keywords)
        
        if is_analytical:
            logger.info(f"📊 [EXECUTE] Processing as analytical query")
            
            # Handle category aggregation specifically
            if ('categor' in instruction_lower or 'product' in instruction_lower) and ('total' in instruction_lower or 'quantit' in instruction_lower or 'sum' in instruction_lower):
                try:
                    # Look for category and quantity columns
                    category_cols = [col for col in df.columns if 'categor' in col.lower()]
                    quantity_cols = [col for col in df.columns if 'quantit' in col.lower() or 'qty' in col.lower()]
                    
                    if not category_cols:
                        # Try other common category column names
                        category_cols = [col for col in df.columns if any(term in col.lower() for term in ['type', 'class', 'group', 'product'])]
                    
                    if not quantity_cols:
                        # Try other common quantity column names
                        quantity_cols = [col for col in df.columns if any(term in col.lower() for term in ['amount', 'count', 'num', 'total', 'sales'])]
                    
                    if category_cols and quantity_cols:
                        category_col = category_cols[0]
                        quantity_col = quantity_cols[0]
                        
                        logger.info(f"📊 [EXECUTE] Aggregating {quantity_col} by {category_col}")
                        
                        # Perform aggregation
                        result_df = df.groupby(category_col)[quantity_col].sum().reset_index()
                        result_dict = result_df.to_dict(orient='records')
                        
                        # Format the result
                        categories_summary = []
                        for row in result_dict:
                            categories_summary.append({
                                "category": row[category_col],
                                "total_quantity": row[quantity_col]
                            })
                        
                        logger.info(f"✅ [EXECUTE] Successfully aggregated {len(categories_summary)} categories")
                        
                        return {
                            "status": "complete",
                            "result": {
                                "categories": categories_summary,
                                "total_categories": len(categories_summary),
                                "summary": f"Found {len(categories_summary)} unique product categories with their total quantities",
                                "columns_used": {
                                    "category_column": category_col,
                                    "quantity_column": quantity_col
                                }
                            },
                            "explanation": f"Successfully scanned the file and found {len(categories_summary)} unique product categories with their total quantities.",
                            "context": {"task_id": task_id}
                        }
                    else:
                        missing = []
                        if not category_cols:
                            missing.append("category column")
                        if not quantity_cols:
                            missing.append("quantity column")
                        
                        return {
                            "status": "error",
                            "error": f"Could not find required columns: {', '.join(missing)}. Available columns: {', '.join(df.columns.tolist())}",
                            "context": {"task_id": task_id, "available_columns": df.columns.tolist()}
                        }
                        
                except Exception as e:
                    logger.error(f"❌ [EXECUTE] Category aggregation failed: {e}", exc_info=True)
                    return {
                        "status": "error",
                        "error": f"Category aggregation failed: {str(e)}",
                        "context": {"task_id": task_id}
                    }
            
            # For other analytical queries, provide basic summary
            try:
                summary = {
                    "file_info": {
                        "rows": len(df),
                        "columns": len(df.columns),
                        "column_names": df.columns.tolist()
                    },
                    "sample_data": df.head(5).to_dict(orient='records'),
                    "instruction_processed": prompt
                }
                
                return {
                    "status": "complete",
                    "result": summary,
                    "explanation": f"Processed analytical query for file with {len(df)} rows and {len(df.columns)} columns",
                    "context": {"task_id": task_id}
                }
                
            except Exception as e:
                logger.error(f"❌ [EXECUTE] Analytical processing failed: {e}", exc_info=True)
                return {
                    "status": "error",
                    "error": f"Analytical processing failed: {str(e)}",
                    "context": {"task_id": task_id}
                }
        
        # Default: provide file summary
        try:
            result = {
                "file_summary": {
                    "file_id": file_id_param,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": df.columns.tolist(),
                    "sample_data": df.head(3).to_dict(orient='records')
                },
                "instruction": prompt
            }
            
            return {
                "status": "complete",
                "result": result,
                "explanation": f"Processed instruction for spreadsheet with {len(df)} rows and {len(df.columns)} columns",
                "context": {"task_id": task_id}
            }
            
        except Exception as e:
            logger.error(f"❌ [EXECUTE] Default processing failed: {e}", exc_info=True)
            return {
                "status": "error",
                "error": f"Processing failed: {str(e)}",
                "context": {"task_id": task_id}
            }
        
    except Exception as e:
        logger.error(f"❌ [EXECUTE] Unexpected error: {e}", exc_info=True)
        return {
            "status": "error",
            "error": f"Unexpected error: {str(e)}",
            "context": {"task_id": message.get('payload', {}).get('task_id', 'unknown') if message else 'unknown'}
        }


@app.post("/continue")
async def continue_action(
    message: Optional[Dict[str, Any]] = None,
    task_id: Optional[str] = Form(None),
    answer: Optional[str] = Form(None)
):
    """
    Resume a paused task with user input.
    
    This endpoint handles continuation of paused tasks from the orchestrator.
    Supports both JSON and form-encoded requests.
    """
    try:
        # Handle both JSON and form-data requests
        if message is None:
            message = {}
        
        # Extract parameters from form data if provided
        if task_id:
            if 'payload' not in message:
                message['payload'] = {}
            message['payload']['task_id'] = task_id
        if answer:
            message['answer'] = answer
        
        task_id_param = message.get('payload', {}).get('task_id')
        user_answer = message.get('answer', '')
        
        if not task_id_param:
            return {
                "status": "error",
                "error": "task_id required in payload",
                "context": {}
            }
            
        logger.info(f"▶️ [CONTINUE] Resuming TaskID={task_id_param} with Answer='{user_answer}'")
        
        # For now, return a simple continuation response
        # In a full implementation, this would resume the specific paused operation
        return {
            "status": "complete",
            "result": {
                "task_id": task_id_param,
                "user_answer": user_answer,
                "message": "Task continuation not fully implemented yet"
            },
            "explanation": f"Received continuation for task {task_id_param}",
            "context": {"task_id": task_id_param}
        }
        
    except Exception as e:
        logger.error(f"❌ [CONTINUE] Failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": f"Continue failed: {str(e)}",
            "context": {"task_id": message.get('payload', {}).get('task_id', 'unknown') if message else 'unknown'}
        }


def _convert_agent_response_to_standard(
    agent_response: Dict[str, Any],
    route: str
) -> StandardResponse:
    """
    Convert AgentResponse format to StandardResponse format.
    
    Args:
        agent_response: Response from spreadsheet_agent
        route: The endpoint route
    
    Returns:
        StandardResponse object
    """
    status = agent_response.get('status')
    
    # Determine success based on status
    success = status in ['complete', 'partial']
    
    # Extract metrics
    metrics_data = agent_response.get('metrics', {})
    metrics = StandardResponseMetrics(
        rows_processed=metrics_data.get('rows_processed', 0),
        columns_affected=metrics_data.get('columns_affected', 0),
        llm_calls=metrics_data.get('llm_calls', 0)
    )
    
    # Build response
    response = StandardResponse(
        success=success,
        route=route,
        task_type=agent_response.get('metadata', {}).get('task_type', 'execute'),
        data=agent_response.get('result', {}),
        message=agent_response.get('explanation', ''),
        metrics=metrics,
        confidence=1.0 if success else 0.0
    )
    
    # Handle NEEDS_INPUT status
    if status == 'needs_input':
        response.needs_clarification = True
        response.data = {
            "question": agent_response.get('question'),
            "question_type": agent_response.get('question_type'),
            "choices": agent_response.get('choices'),
            "context": agent_response.get('context')
        }
    
    # Handle ERROR status
    if status == 'error':
        response.success = False
        response.message = agent_response.get('error', 'Unknown error')
    
    # Handle PARTIAL status
    if status == 'partial':
        response.data = agent_response.get('partial_result', {})
        response.data['progress'] = agent_response.get('progress', 0.0)
    
    return response


@app.get("/stats", response_model=ApiResponse)
async def get_stats():
    """Get agent statistics"""
    try:
        stats = {
            "files_managed": len(file_manager.list_files()),
            "active_conversations": len(getattr(getattr(sys.modules[__name__], '_thread_local', None), 'dataframes_by_thread', {})),
            "cache_stats": spreadsheet_memory.get_cache_stats(),
            "llm_providers": [p['name'] for p in query_agent.providers],
            "storage_dir": str(STORAGE_DIR),
            "version": "2.0.0"
        }
        
        return ApiResponse(success=True, result=stats)
    
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        return ApiResponse(success=False, error=str(e))


@app.get("/metrics", response_model=ApiResponse)
async def get_metrics():
    """Get detailed agent metrics including API calls, timing, LLM calls, and errors"""
    try:
        uptime_seconds = time.time() - call_metrics["start_time"]
        
        # Calculate average response times
        avg_timing = {}
        for endpoint, timings in call_metrics["api_timing"].items():
            if timings:
                avg_timing[endpoint] = round(sum(timings) / len(timings), 3)
        
        metrics = {
            "api_calls": dict(call_metrics["api_calls"]),
            "api_errors": dict(call_metrics["api_errors"]),
            "avg_response_time_seconds": avg_timing,
            "llm_calls": call_metrics["llm_calls"],
            "uptime_seconds": round(uptime_seconds, 2),
            "files_managed": len(file_manager.list_files()),
            "cache_stats": spreadsheet_memory.get_cache_stats()
        }
        
        return ApiResponse(success=True, result=metrics)
    
    except Exception as e:
        logger.error(f"Metrics retrieval failed: {e}")
        return ApiResponse(success=False, error=str(e))


@app.post("/metrics/reset", response_model=ApiResponse)
async def reset_metrics():
    """Reset all metrics counters to zero"""
    try:
        # Store old metrics
        old_metrics = dict(call_metrics)
        
        # Reset counters
        call_metrics["api_calls"] = defaultdict(int)
        call_metrics["api_timing"] = defaultdict(list)
        call_metrics["api_errors"] = defaultdict(int)
        call_metrics["llm_calls"] = {
            "total": 0,
            "by_provider": defaultdict(int)
        }
        call_metrics["start_time"] = time.time()
        
        return ApiResponse(success=True, result={
            "message": "Metrics reset successfully",
            "previous_total_calls": sum(old_metrics.get("api_calls", {}).values())
        })
    
    except Exception as e:
        logger.error(f"Metrics reset failed: {e}")
        return ApiResponse(success=False, error=str(e))


# Main execution
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=AGENT_PORT)
