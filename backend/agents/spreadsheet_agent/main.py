"""
Main FastAPI application for the Spreadsheet Agent.

This module consolidates all API routes and uses the modular components.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from asyncio import Lock as AsyncLock

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Header
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

# Import modular components
from .config import STORAGE_DIR, AGENT_PORT, MAX_FILE_SIZE_MB
from .models import ApiResponse, CreateSpreadsheetRequest, NaturalLanguageQueryRequest
from .memory import spreadsheet_memory
from . import session as session_module  # Import module for _thread_local access
from .session import (
    get_conversation_dataframes,
    get_conversation_file_paths,
    ensure_file_loaded,
    get_dataframe_state,
    store_dataframe,
    get_dataframe
)
from .llm_agent import query_agent
from .code_generator import generate_modification_code, generate_csv_from_instruction
from .display import dataframe_to_canvas, format_dataframe_preview
from .utils import (
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
from .spreadsheet_session_manager import spreadsheet_session_manager

# Create FastAPI app
app = FastAPI(title="Spreadsheet Agent", version="2.0.0")

# Create storage directory
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# Metrics tracking
import time
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

@app.post("/upload", response_model=ApiResponse)
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

        return ApiResponse(success=True, result={
            "file_id": file_id,
            "filename": file.filename,
            "file_path": file_location,
            "rows": len(df),
            "columns": len(df.columns),
            "orchestrator_format": metadata.to_orchestrator_format(),
            "canvas_display": canvas_display
        })

    except Exception as e:
        logger.error(f"File upload failed: {e}", exc_info=True)
        return ApiResponse(success=False, error=str(e))


@app.post("/nl_query", response_model=ApiResponse)
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
                session_history = spreadsheet_session_manager.get_session_history(thread_id, limit=5)
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
        
        # Update dataframe if modified
        if result.final_dataframe is not None:
            store_dataframe(file_id, result.final_dataframe, file_paths.get(file_id, ""), thread_id)
        
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
        return ApiResponse(success=True, result=response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Natural language query failed: {e}", exc_info=True)
        return ApiResponse(success=False, error=str(e))


@app.post("/transform", response_model=ApiResponse)
async def transform_data(
    file_id: str = Form(...),
    instruction: str = Form(...),
    thread_id: Optional[str] = Form(None)
):
    """Transform spreadsheet data using natural language instruction"""
    try:
        thread_id = thread_id or "default"
        
        # Ensure file is loaded
        if not ensure_file_loaded(file_id, thread_id, file_manager):
            raise HTTPException(status_code=404, detail=f"File {file_id} not found")
        
        # Get dataframe
        df = get_dataframe(file_id, thread_id)
        if df is None:
            raise HTTPException(status_code=500, detail="Failed to load dataframe")
        
        # Generate code
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
        
        # Update dataframe
        store_dataframe(file_id, modified_df, file_paths.get(file_id, ""), thread_id)
        
        # Track operation
        if thread_id != "default":
            try:
                spreadsheet_session_manager.track_operation(
                    thread_id=thread_id,
                    file_id=file_id,
                    operation="transform",
                    description=instruction,
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
            metadata={'operation': 'transform', 'instruction': instruction}
        )
        
        return ApiResponse(success=True, result={
            "file_id": file_id,
            "rows": len(modified_df),
            "columns": len(modified_df.columns),
            "code_executed": code,
            "canvas_display": canvas_display
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transform failed: {e}", exc_info=True)
        return ApiResponse(success=False, error=str(e))


@app.post("/get_summary", response_model=ApiResponse)
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
        
        if show_preview:
            canvas_display = dataframe_to_canvas(
                df=df,
                title=f"Preview: {filename}",
                filename=filename,
                display_mode='full',
                max_rows=10,
                file_id=file_id
            )
            summary["canvas_display"] = canvas_display
        
        return ApiResponse(success=True, result=summary)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get summary failed: {e}", exc_info=True)
        return ApiResponse(success=False, error=str(e))


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
        
        return ApiResponse(success=True, result={
            "message": f"Displaying {len(display_df)} rows",
            "canvas_display": canvas_display
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Display failed: {e}", exc_info=True)
        return ApiResponse(success=False, error=str(e))


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
        store_dataframe(file_id, modified_df, file_paths.get(file_id, ""), thread_id)
        
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


@app.post("/create", response_model=ApiResponse)
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
                return ApiResponse(
                    success=False,
                    error="Wrong endpoint: Use /nl_query for analysis/summary tasks, not /create"
                )
        
        # Generate CSV content if needed
        if request.instruction:
            csv_content = await generate_csv_from_instruction(
                request.instruction,
                request.content
            )
            if not csv_content:
                return ApiResponse(success=False, error="Failed to generate CSV")
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
        
        return ApiResponse(success=True, result={
            "file_id": file_id,
            "filename": filename,
            "file_path": str(file_path),
            "rows": len(df),
            "columns": len(df.columns),
            "canvas_display": canvas_display
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Create spreadsheet failed: {e}", exc_info=True)
        return ApiResponse(success=False, error=str(e))


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
