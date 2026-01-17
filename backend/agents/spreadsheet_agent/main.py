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

# Import orchestrator schemas for bidirectional communication
from schemas import AgentResponse, AgentResponseStatus, OrchestratorMessage, DialogueContext

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

# Import DialogueManager for orchestrator communication - use schemas.py AgentResponse
from agents.spreadsheet_agent.dialogue_manager import dialogue_manager

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
    """Get summary of spreadsheet with intelligent parsing and document structure analysis"""
    try:
        thread_id = thread_id or "default"
        
        if not ensure_file_loaded(file_id, thread_id, file_manager):
            raise HTTPException(status_code=404, detail="File not found")
        
        df = get_dataframe(file_id, thread_id)
        if df is None:
            raise HTTPException(status_code=500, detail="Failed to load dataframe")
        
        metadata = file_manager.get_file(file_id)
        filename = metadata.original_name if metadata else "unknown.csv"
        
        # Import the global spreadsheet parser
        from agents.spreadsheet_agent.spreadsheet_parser import spreadsheet_parser
        
        # Perform intelligent parsing
        try:
            parsed_spreadsheet = spreadsheet_parser.parse_dataframe(df, file_id, "Sheet1")
            
            # Get intelligent summary with document structure
            intelligent_summary = spreadsheet_parser.get_metadata_summary(parsed_spreadsheet)
            
            # Build enhanced summary
            summary = {
                "filename": filename,
                "headers": df.columns.tolist(),
                "rows": convert_numpy_types(df.head(5).to_dict(orient="records")),
                "dtypes": {k: str(v) for k, v in df.dtypes.to_dict().items()},
                "total_rows": len(df),
                "total_columns": len(df.columns),
                
                # Intelligent parsing results
                "document_analysis": {
                    "document_type": intelligent_summary["document_type"],
                    "parsing_confidence": intelligent_summary["parsing_confidence"],
                    "sections_detected": intelligent_summary["sections_count"],
                    "tables_detected": intelligent_summary["tables_count"],
                    "has_metadata": intelligent_summary["has_metadata"],
                    "has_line_items": intelligent_summary["has_line_items"],
                    "has_summary": intelligent_summary["has_summary"],
                    "intentional_gaps": intelligent_summary["intentional_gaps"],
                    "metadata_items": intelligent_summary["metadata_items"]
                },
                
                # Primary table information
                "primary_table": None
            }
            
            # Add primary table info if available
            primary_table = spreadsheet_parser.get_primary_table(parsed_spreadsheet)
            if primary_table:
                region, table_df, schema = primary_table
                summary["primary_table"] = {
                    "region": {
                        "start_row": region.start_row,
                        "end_row": region.end_row,
                        "start_col": region.start_col,
                        "end_col": region.end_col,
                        "confidence": region.confidence
                    },
                    "schema": {
                        "headers": schema.headers,
                        "dtypes": schema.dtypes,
                        "row_count": schema.row_count,
                        "col_count": schema.col_count
                    }
                }
            
            # Add extracted metadata if available
            if parsed_spreadsheet.metadata:
                summary["extracted_metadata"] = parsed_spreadsheet.metadata
                
        except Exception as e:
            logger.warning(f"Intelligent parsing failed for {file_id}: {e}")
            # Fallback to basic summary
            summary = {
                "filename": filename,
                "headers": df.columns.tolist(),
                "rows": convert_numpy_types(df.head(5).to_dict(orient="records")),
                "dtypes": {k: str(v) for k, v in df.dtypes.to_dict().items()},
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "document_analysis": {
                    "parsing_error": str(e),
                    "fallback_mode": True
                }
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
            data=convert_numpy_types(summary),
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
    request: Request
):
    """
    Unified execution endpoint supporting orchestrator communication.
    
    This endpoint handles requests from the orchestrator and returns AgentResponse format.
    Supports both JSON and form-encoded requests.
    """
    start_time = time.time()
    
    try:
        # Handle both JSON and form-data requests
        content_type = request.headers.get("content-type", "")
        
        if "application/json" in content_type:
            # JSON request - parse OrchestratorMessage
            body = await request.json()
            
            # Validate OrchestratorMessage format
            try:
                orchestrator_msg = OrchestratorMessage(**body)
            except Exception as e:
                logger.error(f"❌ [EXECUTE] Invalid OrchestratorMessage format: {e}")
                return AgentResponse(
                    status=AgentResponseStatus.ERROR,
                    error=f"Invalid message format: {str(e)}"
                ).model_dump()
        else:
            # Form data request - build OrchestratorMessage from form fields
            form_data = await request.form()
            
            # Extract form fields
            action = form_data.get('action')
            prompt = form_data.get('prompt') or form_data.get('instruction')
            file_id = form_data.get('file_id')
            thread_id = form_data.get('thread_id', 'default')
            
            # Build OrchestratorMessage
            orchestrator_msg = OrchestratorMessage(
                action=action,
                prompt=prompt,
                payload={
                    'file_id': file_id,
                    'thread_id': thread_id
                },
                source="orchestrator",
                target="spreadsheet_agent"
            )
        
        # Extract parameters
        payload = orchestrator_msg.payload or {}
        action = orchestrator_msg.action
        prompt = orchestrator_msg.prompt
        file_id_param = payload.get('file_id')
        thread_id_param = payload.get('thread_id', 'default')
        
        # Generate task_id
        task_id = payload.get('task_id', f"task-{int(time.time())}")
        
        logger.info(f"🚀 [EXECUTE] Action={action}, Prompt={prompt[:100] if prompt else 'None'}..., TaskID={task_id}")
        
        # Validate required fields
        if not file_id_param:
            logger.error("❌ [EXECUTE] Missing file_id in payload")
            return AgentResponse(
                status=AgentResponseStatus.ERROR,
                error="Missing required field: file_id",
                context={"task_id": task_id}
            ).model_dump()
        
        if not prompt and not action:
            logger.error("❌ [EXECUTE] Missing instruction/prompt and action in payload")
            return AgentResponse(
                status=AgentResponseStatus.ERROR,
                error="Either 'action' or 'prompt' must be provided",
                context={"task_id": task_id}
            ).model_dump()
        
        logger.info(f"📁 [EXECUTE] Processing: file_id={file_id_param}, thread_id={thread_id_param}")
        
        # Ensure file is loaded
        if not ensure_file_loaded(file_id_param, thread_id_param, file_manager):
            logger.error(f"❌ [EXECUTE] File not found: {file_id_param}")
            return AgentResponse(
                status=AgentResponseStatus.ERROR,
                error=f"File {file_id_param} not found or could not be loaded",
                context={"task_id": task_id, "file_id": file_id_param}
            ).model_dump()
        
        # Get dataframe
        df = get_dataframe(file_id_param, thread_id_param)
        if df is None:
            logger.error(f"❌ [EXECUTE] Failed to load dataframe: {file_id_param}")
            return AgentResponse(
                status=AgentResponseStatus.ERROR,
                error=f"Failed to load dataframe for file {file_id_param}",
                context={"task_id": task_id, "file_id": file_id_param}
            ).model_dump()
        
        logger.info(f"📊 [EXECUTE] Dataframe loaded: {len(df)} rows × {len(df.columns)} cols")
        
        # Route based on action or process prompt
        if action:
            # Action-based routing
            if action == "/get_summary" or action == "get_summary":
                # Route to get_summary endpoint
                try:
                    summary_response = await get_summary(
                        file_id=file_id_param,
                        show_preview=True,
                        thread_id=thread_id_param
                    )
                    
                    # Create metrics
                    metrics = dialogue_manager.create_metrics(
                        start_time=start_time,
                        rows_processed=len(df),
                        columns_affected=len(df.columns)
                    )
                    
                    if summary_response.success:
                        return AgentResponse(
                            status=AgentResponseStatus.COMPLETE,
                            result=summary_response.data,
                            context={"task_id": task_id, "action": action}
                        ).model_dump()
                    else:
                        return AgentResponse(
                            status=AgentResponseStatus.ERROR,
                            error=summary_response.message,
                            context={"task_id": task_id, "action": action}
                        ).model_dump()
                        
                except Exception as e:
                    logger.error(f"❌ [EXECUTE] get_summary failed: {e}", exc_info=True)
                    return AgentResponse(
                        status=AgentResponseStatus.ERROR,
                        error=f"Summary generation failed: {str(e)}",
                        context={"task_id": task_id, "action": action}
                    ).model_dump()
            
            elif action == "/analyze_structure" or action == "analyze_structure":
                # Route to intelligent structure analysis
                try:
                    # Import the global spreadsheet parser
                    from agents.spreadsheet_agent.spreadsheet_parser import spreadsheet_parser
                    
                    # Perform intelligent parsing
                    parsed_spreadsheet = spreadsheet_parser.parse_dataframe(df, file_id_param, "Sheet1")
                    
                    # Get intelligent analysis
                    intelligent_summary = spreadsheet_parser.get_metadata_summary(parsed_spreadsheet)
                    
                    # Build comprehensive structure info
                    structure_info = {
                        "file_id": file_id_param,
                        "basic_info": {
                            "shape": {"rows": len(df), "columns": len(df.columns)},
                            "columns": df.columns.tolist(),
                            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                            "null_counts": df.isnull().sum().to_dict()
                        },
                        "document_analysis": {
                            "document_type": intelligent_summary["document_type"],
                            "parsing_confidence": intelligent_summary["parsing_confidence"],
                            "sections_detected": intelligent_summary["sections_count"],
                            "tables_detected": intelligent_summary["tables_count"],
                            "has_metadata": intelligent_summary["has_metadata"],
                            "has_line_items": intelligent_summary["has_line_items"],
                            "has_summary": intelligent_summary["has_summary"],
                            "intentional_gaps": intelligent_summary["intentional_gaps"],
                            "metadata_items": intelligent_summary["metadata_items"]
                        },
                        "sections": [],
                        "tables": [],
                        "extracted_metadata": parsed_spreadsheet.metadata
                    }
                    
                    # Add section details
                    for section in parsed_spreadsheet.sections:
                        structure_info["sections"].append({
                            "type": section.section_type.value,
                            "content_type": section.content_type.value,
                            "start_row": section.start_row,
                            "end_row": section.end_row,
                            "row_count": section.row_count,
                            "confidence": section.confidence,
                            "metadata": section.metadata
                        })
                    
                    # Add table details
                    for region, table_df, schema in parsed_spreadsheet.tables:
                        structure_info["tables"].append({
                            "region": {
                                "start_row": region.start_row,
                                "end_row": region.end_row,
                                "start_col": region.start_col,
                                "end_col": region.end_col,
                                "confidence": region.confidence,
                                "size": region.size
                            },
                            "schema": {
                                "headers": schema.headers,
                                "dtypes": schema.dtypes,
                                "row_count": schema.row_count,
                                "col_count": schema.col_count,
                                "numeric_columns": schema.get_numeric_columns(),
                                "text_columns": schema.get_text_columns(),
                                "date_columns": schema.get_date_columns()
                            },
                            "sample_data": convert_numpy_types(table_df.head(3).to_dict(orient='records')) if not table_df.empty else []
                        })
                    
                    # Create metrics
                    metrics = dialogue_manager.create_metrics(
                        start_time=start_time,
                        rows_processed=len(df),
                        columns_affected=len(df.columns)
                    )
                    
                    return AgentResponse(
                        status=AgentResponseStatus.COMPLETE,
                        result=structure_info,
                        context={"task_id": task_id, "action": action, "parsing_confidence": intelligent_summary['parsing_confidence']}
                    ).model_dump()
                    
                except Exception as e:
                    logger.error(f"❌ [EXECUTE] analyze_structure failed: {e}", exc_info=True)
                    # Fallback to basic structure analysis
                    try:
                        structure_info = {
                            "file_id": file_id_param,
                            "basic_info": {
                                "shape": {"rows": len(df), "columns": len(df.columns)},
                                "columns": df.columns.tolist(),
                                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                                "sample_data": convert_numpy_types(df.head(3).to_dict(orient='records')),
                                "null_counts": df.isnull().sum().to_dict()
                            },
                            "parsing_error": str(e),
                            "fallback_mode": True
                        }
                        
                        metrics = dialogue_manager.create_metrics(
                            start_time=start_time,
                            rows_processed=len(df),
                            columns_affected=len(df.columns)
                        )
                        
                        return AgentResponse(
                            status=AgentResponseStatus.COMPLETE,
                            result=structure_info,
                            context={"task_id": task_id, "action": action, "fallback_mode": True}
                        ).model_dump()
                        
                    except Exception as fallback_error:
                        return AgentResponse(
                            status=AgentResponseStatus.ERROR,
                            error=f"Structure analysis failed: {str(fallback_error)}",
                            context={"task_id": task_id, "action": action}
                        ).model_dump()
            
            elif action == "/detect_anomalies" or action == "detect_anomalies":
                # Route to anomaly detection with orchestrator integration
                try:
                    # Import anomaly detector
                    from agents.spreadsheet_agent.anomaly_detector import AnomalyDetector
                    
                    # Create detector instance
                    anomaly_detector = AnomalyDetector()
                    
                    # Detect anomalies
                    anomalies = anomaly_detector.detect_anomalies(df)
                    
                    logger.info(f"🔍 [EXECUTE] Detected {len(anomalies)} anomalies")
                    
                    # If anomalies found, return NEEDS_INPUT for user decision
                    if anomalies:
                        # Build choices for the first anomaly (handle one at a time)
                        first_anomaly = anomalies[0]
                        
                        choices = []
                        for fix in first_anomaly.suggested_fixes:
                            choices.append({
                                "id": fix.action,
                                "label": fix.description,
                                "safe": fix.safe,
                                "parameters": fix.parameters
                            })
                        
                        # Store anomaly context for continuation
                        anomaly_context = {
                            "anomalies": [anomaly.to_dict() for anomaly in anomalies],
                            "current_anomaly_index": 0,
                            "file_id": file_id_param,
                            "thread_id": thread_id_param
                        }
                        
                        # Store in dialogue manager for continuation
                        dialogue_manager.store_pending_question(
                            task_id=task_id,
                            question=first_anomaly.message,
                            question_type="choice",
                            choices=choices,
                            context=anomaly_context
                        )
                        
                        return AgentResponse(
                            status=AgentResponseStatus.NEEDS_INPUT,
                            question=first_anomaly.message,
                            question_type="choice",
                            options=[choice["id"] for choice in choices],
                            context={
                                "task_id": task_id, 
                                "anomaly_type": first_anomaly.type,
                                "choices": choices,
                                "anomalies_count": len(anomalies),
                                "current_anomaly": first_anomaly.type
                            }
                        ).model_dump()
                    
                    else:
                        # No anomalies found
                        metrics = dialogue_manager.create_metrics(
                            start_time=start_time,
                            rows_processed=len(df),
                            columns_affected=len(df.columns)
                        )
                        
                        return AgentResponse(
                            status=AgentResponseStatus.COMPLETE,
                            result={
                                "anomalies_detected": 0,
                                "data_quality": "good",
                                "message": "No data quality issues detected"
                            },
                            context={"task_id": task_id, "action": action}
                        ).model_dump()
                        
                except Exception as e:
                    logger.error(f"❌ [EXECUTE] detect_anomalies failed: {e}", exc_info=True)
                    return AgentResponse(
                        status=AgentResponseStatus.ERROR,
                        error=f"Anomaly detection failed: {str(e)}",
                        context={"task_id": task_id, "action": action}
                    ).model_dump()
            
            elif action == "/execute_plan" or action == "execute_plan":
                # Route to multi-step plan execution
                try:
                    # Import planner
                    from agents.spreadsheet_agent.planner import planner
                    
                    # Get plan parameters from payload
                    plan_instruction = payload.get('plan_instruction') or prompt
                    if not plan_instruction:
                        return AgentResponse(
                            status=AgentResponseStatus.ERROR,
                            error="plan_instruction required for execute_plan action",
                            context={"task_id": task_id, "action": action}
                        ).model_dump()
                    
                    logger.info(f"📋 [EXECUTE] Multi-step planning for: {plan_instruction[:100]}...")
                    
                    # Generate DataFrame context
                    df_context = {
                        "shape": df.shape,
                        "columns": df.columns.tolist(),
                        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                        "sample": df.head(3).to_dict(orient='records')
                    }
                    
                    # Propose plan
                    plan = await planner.propose_plan(df, plan_instruction, df_context)
                    
                    logger.info(f"📋 [EXECUTE] Generated plan with {len(plan.actions)} actions")
                    
                    # Simulate plan first
                    simulation_result = planner.simulate_plan(plan, df)
                    
                    if not simulation_result["success"]:
                        # Simulation failed - ask user if they want to proceed anyway
                        simulation_warnings = simulation_result.get("warnings", [])
                        simulation_errors = [step.get("error") for step in simulation_result.get("simulation_log", []) if step.get("error")]
                        
                        choices = [
                            {
                                "id": "proceed_anyway",
                                "label": "Execute plan despite simulation warnings",
                                "safe": False
                            },
                            {
                                "id": "cancel_plan",
                                "label": "Cancel plan execution",
                                "safe": True
                            }
                        ]
                        
                        # Store plan context for continuation
                        plan_context = {
                            "plan_id": plan.plan_id,
                            "file_id": file_id_param,
                            "thread_id": thread_id_param,
                            "simulation_result": simulation_result
                        }
                        
                        dialogue_manager.store_pending_question(
                            task_id=task_id,
                            question=f"Plan simulation detected issues: {'; '.join(simulation_errors[:2])}. Do you want to proceed anyway?",
                            question_type="confirmation",
                            choices=choices,
                            context=plan_context
                        )
                        
                        return AgentResponse(
                            status=AgentResponseStatus.NEEDS_INPUT,
                            question=f"Plan simulation detected issues: {'; '.join(simulation_errors[:2])}. Do you want to proceed anyway?",
                            question_type="confirmation",
                            options=["proceed_anyway", "cancel_plan"],
                            context={"task_id": task_id, "plan_id": plan.plan_id}
                        ).model_dump()
                    
                    # Simulation successful - execute plan
                    modified_df, execution_result = planner.execute_plan(plan, df, force=False)
                    
                    if execution_result["success"]:
                        # Update dataframe in session
                        thread_paths = get_conversation_file_paths(thread_id_param)
                        store_dataframe(file_id_param, modified_df, thread_paths.get(file_id_param, file_paths.get(file_id_param, "")), thread_id_param)
                        
                        # Create metrics
                        metrics = dialogue_manager.create_metrics(
                            start_time=start_time,
                            rows_processed=len(modified_df),
                            columns_affected=len(modified_df.columns)
                        )
                        
                        return AgentResponse(
                            status=AgentResponseStatus.COMPLETE,
                            result={
                                "plan_id": plan.plan_id,
                                "actions_executed": execution_result["actions_executed"],
                                "final_shape": {"rows": len(modified_df), "columns": len(modified_df.columns)},
                                "execution_log": execution_result["execution_log"],
                                "plan_reasoning": plan.reasoning,
                                "message": f"Multi-step plan executed successfully: {execution_result['actions_executed']} actions completed"
                            },
                            context={"task_id": task_id, "action": action, "plan_id": plan.plan_id}
                        ).model_dump()
                    
                    else:
                        # Execution failed
                        return AgentResponse(
                            status=AgentResponseStatus.ERROR,
                            error=f"Plan execution failed: {execution_result.get('error', 'Unknown error')}",
                            context={
                                "task_id": task_id,
                                "action": action,
                                "plan_id": plan.plan_id,
                                "execution_log": execution_result.get("execution_log", [])
                            }
                        ).model_dump()
                        
                except Exception as e:
                    logger.error(f"❌ [EXECUTE] execute_plan failed: {e}", exc_info=True)
                    return AgentResponse(
                        status=AgentResponseStatus.ERROR,
                        error=f"Multi-step plan execution failed: {str(e)}",
                        context={"task_id": task_id, "action": action}
                    ).model_dump()
            
            else:
                # Unknown action
                return AgentResponse(
                    status=AgentResponseStatus.ERROR,
                    error=f"Unknown action: {action}",
                    context={"task_id": task_id, "action": action}
                ).model_dump()
        
        elif prompt:
            # Prompt-based processing
            try:
                # Use natural language query processing
                from agents.spreadsheet_agent.models import NaturalLanguageQueryRequest
                
                nl_request = NaturalLanguageQueryRequest(
                    file_id=file_id_param,
                    question=prompt,
                    max_iterations=3
                )
                
                logger.info(f"🤖 [EXECUTE] Processing prompt with natural language agent")
                nl_response = await natural_language_query(nl_request, thread_id_param)
                
                # Create metrics
                metrics = dialogue_manager.create_metrics(
                    start_time=start_time,
                    rows_processed=len(df),
                    columns_affected=len(df.columns),
                    llm_calls=nl_response.metrics.llm_calls if nl_response.metrics else 0
                )
                
                # Convert StandardResponse to AgentResponse
                if nl_response.success:
                    return AgentResponse(
                        status=AgentResponseStatus.COMPLETE,
                        result=nl_response.data,
                        context={"task_id": task_id, "prompt": prompt[:100]}
                    ).model_dump()
                else:
                    return AgentResponse(
                        status=AgentResponseStatus.ERROR,
                        error=nl_response.message,
                        context={"task_id": task_id, "prompt": prompt[:100]}
                    ).model_dump()
                    
            except Exception as e:
                logger.error(f"❌ [EXECUTE] Prompt processing failed: {e}", exc_info=True)
                return AgentResponse(
                    status=AgentResponseStatus.ERROR,
                    error=f"Prompt processing failed: {str(e)}",
                    context={"task_id": task_id, "prompt": prompt[:100]}
                ).model_dump()
        
        # Should not reach here
        return AgentResponse(
            status=AgentResponseStatus.ERROR,
            error="No valid action or prompt provided",
            context={"task_id": task_id}
        ).model_dump()
        
    except Exception as e:
        logger.error(f"❌ [EXECUTE] Unexpected error: {e}", exc_info=True)
        return AgentResponse(
            status=AgentResponseStatus.ERROR,
            error=f"Unexpected error: {str(e)}",
            context={"task_id": "unknown"}
        ).model_dump()


@app.post("/continue")
async def continue_action(
    request: Request
):
    """
    Resume a paused task with user input.
    
    This endpoint handles continuation of paused tasks from the orchestrator.
    Supports both JSON and form-encoded requests.
    """
    start_time = time.time()
    
    try:
        # Handle both JSON and form-data requests
        content_type = request.headers.get("content-type", "")
        
        if "application/json" in content_type:
            # JSON request - parse OrchestratorMessage
            body = await request.json()
            
            # Validate OrchestratorMessage format
            try:
                orchestrator_msg = OrchestratorMessage(**body)
            except Exception as e:
                logger.error(f"❌ [CONTINUE] Invalid OrchestratorMessage format: {e}")
                return AgentResponse(
                    status=AgentResponseStatus.ERROR,
                    error=f"Invalid message format: {str(e)}"
                ).model_dump()
        else:
            # Form data request - build OrchestratorMessage from form fields
            form_data = await request.form()
            
            # Extract form fields
            task_id = form_data.get('task_id')
            answer = form_data.get('answer')
            
            # Build OrchestratorMessage
            orchestrator_msg = OrchestratorMessage(
                type="continue",
                payload={'task_id': task_id},
                answer=answer,
                source="orchestrator",
                target="spreadsheet_agent"
            )
        
        # Extract parameters
        payload = orchestrator_msg.payload or {}
        task_id_param = payload.get('task_id')
        user_answer = orchestrator_msg.answer or ""
        
        if not task_id_param:
            logger.error("❌ [CONTINUE] Missing task_id in payload")
            return AgentResponse(
                status=AgentResponseStatus.ERROR,
                error="task_id required in payload",
                context={}
            ).model_dump()
            
        logger.info(f"▶️ [CONTINUE] Resuming TaskID={task_id_param} with Answer='{user_answer}'")
        
        # Check if we have a pending question for this task
        pending_question = dialogue_manager.get_pending_question(task_id_param)
        if not pending_question:
            logger.warning(f"⚠️ [CONTINUE] No pending question found for task {task_id_param}")
            # For now, return a simple continuation response
            return AgentResponse(
                status=AgentResponseStatus.COMPLETE,
                result={
                    "task_id": task_id_param,
                    "user_answer": user_answer,
                    "message": "Task continuation completed (no pending question found)"
                },
                explanation=f"Received continuation for task {task_id_param}",
                context={"task_id": task_id_param}
            ).model_dump()
        
        # Load dialogue state
        dialogue_state = dialogue_manager.load_state(task_id_param)
        
        # Handle different question types
        question_type = pending_question.get("question_type")
        context = pending_question.get("context", {})
        
        if question_type == "choice":
            # Handle anomaly fix continuation (and other choice-based questions)
            try:
                # Get anomaly context
                anomalies_data = context.get("anomalies", [])
                current_index = context.get("current_anomaly_index", 0)
                file_id = context.get("file_id")
                thread_id = context.get("thread_id", "default")
                
                if not anomalies_data or current_index >= len(anomalies_data):
                    return AgentResponse(
                        status=AgentResponseStatus.ERROR,
                        error="Invalid anomaly context",
                        context={"task_id": task_id_param}
                    ).model_dump()
                
                # Load dataframe
                if not ensure_file_loaded(file_id, thread_id, file_manager):
                    return AgentResponse(
                        status=AgentResponseStatus.ERROR,
                        error=f"File {file_id} not found",
                        context={"task_id": task_id_param}
                    ).model_dump()
                
                df = get_dataframe(file_id, thread_id)
                if df is None:
                    return AgentResponse(
                        status=AgentResponseStatus.ERROR,
                        error=f"Failed to load dataframe for file {file_id}",
                        context={"task_id": task_id_param}
                    ).model_dump()
                
                # Parse user answer (should be fix action ID)
                selected_fix_action = user_answer.strip()
                
                # Get current anomaly
                current_anomaly_data = anomalies_data[current_index]
                
                # Find the selected fix
                selected_fix = None
                for fix_data in current_anomaly_data.get("suggested_fixes", []):
                    if fix_data.get("action") == selected_fix_action:
                        selected_fix = fix_data
                        break
                
                if not selected_fix:
                    return AgentResponse(
                        status=AgentResponseStatus.ERROR,
                        error=f"Invalid fix selection: {selected_fix_action}",
                        context={"task_id": task_id_param}
                    ).model_dump()
                
                # Apply the fix
                from agents.spreadsheet_agent.anomaly_detector import AnomalyDetector, Anomaly, AnomalyFix
                
                # Reconstruct anomaly and fix objects
                anomaly = Anomaly(
                    type=current_anomaly_data["type"],
                    columns=current_anomaly_data["columns"],
                    sample_values=current_anomaly_data["sample_values"],
                    suggested_fixes=[],  # Not needed for apply_fix
                    severity=current_anomaly_data["severity"],
                    message=current_anomaly_data["message"],
                    metadata=current_anomaly_data["metadata"]
                )
                
                fix = AnomalyFix(
                    action=selected_fix["action"],
                    description=selected_fix["description"],
                    safe=selected_fix["safe"],
                    parameters=selected_fix["parameters"]
                )
                
                # Apply fix
                detector = AnomalyDetector()
                modified_df = detector.apply_fix(df, anomaly, fix)
                
                # Update dataframe in session
                thread_paths = get_conversation_file_paths(thread_id)
                store_dataframe(file_id, modified_df, thread_paths.get(file_id, file_paths.get(file_id, "")), thread_id)
                
                # Check if there are more anomalies to handle
                next_index = current_index + 1
                if next_index < len(anomalies_data):
                    # More anomalies to handle - ask about next one
                    next_anomaly_data = anomalies_data[next_index]
                    
                    choices = []
                    for fix_data in next_anomaly_data.get("suggested_fixes", []):
                        choices.append({
                            "id": fix_data["action"],
                            "label": fix_data["description"],
                            "safe": fix_data["safe"],
                            "parameters": fix_data["parameters"]
                        })
                    
                    # Update context for next anomaly
                    updated_context = context.copy()
                    updated_context["current_anomaly_index"] = next_index
                    
                    # Store next question
                    dialogue_manager.store_pending_question(
                        task_id=task_id_param,
                        question=next_anomaly_data["message"],
                        question_type="choice",
                        choices=choices,
                        context=updated_context
                    )
                    
                    return AgentResponse(
                        status=AgentResponseStatus.NEEDS_INPUT,
                        question=next_anomaly_data["message"],
                        question_type="choice",
                        options=[choice["id"] for choice in choices],
                        context={"task_id": task_id_param, "anomaly_type": next_anomaly_data["type"]}
                    ).model_dump()
                
                else:
                    # All anomalies handled - return completion
                    metrics = dialogue_manager.create_metrics(
                        start_time=start_time,
                        rows_processed=len(modified_df),
                        columns_affected=len(modified_df.columns)
                    )
                    
                    return AgentResponse(
                        status=AgentResponseStatus.COMPLETE,
                        result={
                            "task_id": task_id_param,
                            "anomalies_fixed": len(anomalies_data),
                            "final_shape": {"rows": len(modified_df), "columns": len(modified_df.columns)},
                            "last_fix_applied": selected_fix["description"],
                            "message": f"All {len(anomalies_data)} anomalies have been resolved"
                        },
                        context={"task_id": task_id_param}
                    ).model_dump()
                    
            except Exception as e:
                logger.error(f"❌ [CONTINUE] Anomaly fix failed: {e}", exc_info=True)
                return AgentResponse(
                    status=AgentResponseStatus.ERROR,
                    error=f"Anomaly fix failed: {str(e)}",
                    context={"task_id": task_id_param}
                ).model_dump()
        
        elif question_type == "confirmation":
            # Handle plan execution confirmation (and other confirmation questions)
            try:
                # Get plan context
                plan_id = context.get("plan_id")
                file_id = context.get("file_id")
                thread_id = context.get("thread_id", "default")
                simulation_result = context.get("simulation_result", {})
                
                if not plan_id:
                    return AgentResponse(
                        status=AgentResponseStatus.ERROR,
                        error="Invalid plan context",
                        context={"task_id": task_id_param}
                    ).model_dump()
                
                # Load dataframe
                if not ensure_file_loaded(file_id, thread_id, file_manager):
                    return AgentResponse(
                        status=AgentResponseStatus.ERROR,
                        error=f"File {file_id} not found",
                        context={"task_id": task_id_param}
                    ).model_dump()
                
                df = get_dataframe(file_id, thread_id)
                if df is None:
                    return AgentResponse(
                        status=AgentResponseStatus.ERROR,
                        error=f"Failed to load dataframe for file {file_id}",
                        context={"task_id": task_id_param}
                    ).model_dump()
                
                # Parse user answer
                user_choice = user_answer.strip()
                
                if user_choice == "cancel_plan":
                    # User cancelled plan execution
                    return AgentResponse(
                        status=AgentResponseStatus.COMPLETE,
                        result={
                            "task_id": task_id_param,
                            "plan_id": plan_id,
                            "action": "cancelled",
                            "message": "Plan execution cancelled by user"
                        },
                        context={"task_id": task_id_param}
                    ).model_dump()
                
                elif user_choice == "proceed_anyway":
                    # User wants to proceed despite warnings
                    from agents.spreadsheet_agent.planner import planner
                    
                    # Get the plan
                    plan = planner.history.get_plan(plan_id)
                    if not plan:
                        return AgentResponse(
                            status=AgentResponseStatus.ERROR,
                            error=f"Plan {plan_id} not found",
                            context={"task_id": task_id_param}
                        ).model_dump()
                    
                    # Execute plan with force=True
                    modified_df, execution_result = planner.execute_plan(plan, df, force=True)
                    
                    if execution_result["success"]:
                        # Update dataframe in session
                        thread_paths = get_conversation_file_paths(thread_id)
                        store_dataframe(file_id, modified_df, thread_paths.get(file_id, file_paths.get(file_id, "")), thread_id)
                        
                        # Create metrics
                        metrics = dialogue_manager.create_metrics(
                            start_time=start_time,
                            rows_processed=len(modified_df),
                            columns_affected=len(modified_df.columns)
                        )
                        
                        return AgentResponse(
                            status=AgentResponseStatus.COMPLETE,
                            result={
                                "task_id": task_id_param,
                                "plan_id": plan_id,
                                "actions_executed": execution_result["actions_executed"],
                                "final_shape": {"rows": len(modified_df), "columns": len(modified_df.columns)},
                                "execution_log": execution_result["execution_log"],
                                "forced_execution": True,
                                "message": f"Plan executed with force: {execution_result['actions_executed']} actions completed despite warnings"
                            },
                            context={"task_id": task_id_param}
                        ).model_dump()
                    
                    else:
                        # Execution failed even with force
                        return AgentResponse(
                            status=AgentResponseStatus.ERROR,
                            error=f"Plan execution failed even with force: {execution_result.get('error', 'Unknown error')}",
                            context={
                                "task_id": task_id_param,
                                "plan_id": plan_id,
                                "execution_log": execution_result.get("execution_log", [])
                            }
                        ).model_dump()
                
                else:
                    return AgentResponse(
                        status=AgentResponseStatus.ERROR,
                        error=f"Invalid choice: {user_choice}",
                        context={"task_id": task_id_param}
                    ).model_dump()
                    
            except Exception as e:
                logger.error(f"❌ [CONTINUE] Plan execution confirmation failed: {e}", exc_info=True)
                return AgentResponse(
                    status=AgentResponseStatus.ERROR,
                    error=f"Plan execution confirmation failed: {str(e)}",
                    context={"task_id": task_id_param}
                ).model_dump()
        
        # Create metrics
        metrics = dialogue_manager.create_metrics(
            start_time=start_time,
            llm_calls=0,
            cache_hits=1  # State was cached
        )
        
        # Clear the pending question
        dialogue_manager.clear_pending_question(task_id_param)
        
        # For other question types, return a simple continuation response
        return AgentResponse(
            status=AgentResponseStatus.COMPLETE,
            result={
                "task_id": task_id_param,
                "user_answer": user_answer,
                "previous_question": pending_question,
                "dialogue_state": dialogue_state,
                "message": "Task continuation completed successfully"
            },
            context={"task_id": task_id_param}
        ).model_dump()
        
    except Exception as e:
        logger.error(f"❌ [CONTINUE] Failed: {e}", exc_info=True)
        return AgentResponse(
            status=AgentResponseStatus.ERROR,
            error=f"Continue failed: {str(e)}",
            context={"task_id": "unknown"}
        ).model_dump()


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


@app.get("/performance/report", response_model=ApiResponse)
async def get_performance_report():
    """Get comprehensive performance report with advanced optimizations"""
    try:
        # Get basic performance report
        report = {
            "basic_metrics": {
                "uptime_seconds": time.time() - call_metrics["start_time"],
                "total_api_calls": sum(call_metrics["api_calls"].values()),
                "total_errors": sum(call_metrics["api_errors"].values()),
                "cache_stats": spreadsheet_memory.get_cache_stats()
            }
        }
        
        # Add advanced performance metrics if available
        try:
            from agents.spreadsheet_agent.performance_optimizer import (
                performance_monitor,
                memory_optimizer,
                advanced_cache
            )
            
            report["advanced_metrics"] = {
                "performance_monitor": performance_monitor.get_performance_report(),
                "memory_optimizer": {
                    "total_session_memory_mb": memory_optimizer.get_total_memory_usage(),
                    "system_memory_info": memory_optimizer.get_system_memory_info(),
                    "should_cleanup": memory_optimizer.should_trigger_cleanup()
                },
                "advanced_cache_stats": {
                    "metadata_cache": advanced_cache.get_stats() if hasattr(advanced_cache, 'get_stats') else {},
                }
            }
            
            # Get spreadsheet parser performance stats
            try:
                from agents.spreadsheet_agent.spreadsheet_parser import spreadsheet_parser
                parser_stats = spreadsheet_parser.get_performance_stats()
                report["advanced_metrics"]["parser_stats"] = parser_stats
            except Exception as e:
                logger.warning(f"Could not get parser stats: {e}")
            
            report["performance_optimizations_enabled"] = True
            
        except ImportError:
            report["performance_optimizations_enabled"] = False
            report["message"] = "Advanced performance optimizations not available"
        
        return ApiResponse(success=True, result=report)
    
    except Exception as e:
        logger.error(f"Performance report failed: {e}")
        return ApiResponse(success=False, error=str(e))


@app.post("/performance/optimize", response_model=ApiResponse)
async def trigger_performance_optimization():
    """Manually trigger performance optimizations"""
    try:
        optimizations_applied = []
        
        # Clear old caches
        spreadsheet_memory.clear_all()
        optimizations_applied.append("Cleared all caches")
        
        # Force garbage collection if advanced optimizations available
        try:
            from agents.spreadsheet_agent.performance_optimizer import memory_optimizer
            
            if memory_optimizer.should_trigger_cleanup():
                memory_optimizer.force_garbage_collection()
                optimizations_applied.append("Forced garbage collection")
            
            # Get memory info after optimization
            memory_info = memory_optimizer.get_system_memory_info()
            
        except ImportError:
            import gc
            gc.collect()
            optimizations_applied.append("Basic garbage collection")
            memory_info = {"basic_gc_only": True}
        
        return ApiResponse(success=True, result={
            "optimizations_applied": optimizations_applied,
            "memory_info_after": memory_info
        })
    
    except Exception as e:
        logger.error(f"Performance optimization failed: {e}")
        return ApiResponse(success=False, error=str(e))


# Main execution
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=AGENT_PORT)
