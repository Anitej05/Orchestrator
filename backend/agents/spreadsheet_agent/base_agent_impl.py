"""
Spreadsheet Agent v3.0 - Complete BaseAgent Implementation

Production-grade spreadsheet analysis with full feature set:
- All data operations (load, process, filter, sort, aggregate, merge, transform)
- Column management (add, drop, rename, fill_na)
- Canvas display integration
- File upload handling
- Streaming support
- Export to multiple formats
"""

import logging
import os
import re
import json
import uuid
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np

from backend.agents.base import BaseAgent, AgentServices, AgentConfig
from backend.agents.base.types import AgentRequest, AgentResponse, ExecutionContext
from backend.agents.base.capability import capability, ParameterSchema

from .config import logger, STORAGE_DIR, AGENT_VERSION
from .client import DataFrameClient, SmartDataResolver
from .llm import LLMClient
from .state import session_state, Session
from .agent_schemas import StepResult, TaskStatus, ExecuteResponse

# Try to import CanvasService
try:
    from backend.services.canvas_service import CanvasService

    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False

logger = logging.getLogger("agents.spreadsheet_agent")


@dataclass
class SpreadsheetAgentConfig(AgentConfig):
    """Configuration for Spreadsheet Agent."""

    max_file_size_mb: int = 100
    max_rows_display: int = 1000
    max_cols_display: int = 50
    enable_caching: bool = True
    cache_ttl_hours: float = 2.0
    max_retries: int = 3
    enable_streaming: bool = True
    max_steps_per_task: int = 20


class SpreadsheetAgent(BaseAgent):
    """
    Production-grade spreadsheet analysis agent with complete feature set.

    Features:
    - Load and save CSV, Excel, JSON files
    - LLM-powered data processing and analysis
    - Filter, sort, aggregate, merge operations
    - Column management (add, drop, rename, fill_na)
    - Data transformation
    - Canvas display for frontend
    - Export to multiple formats
    - File upload handling
    """

    def __init__(
        self,
        agent_id: str = "spreadsheet_agent",
        agent_name: str = "Spreadsheet Agent",
        services: Optional[AgentServices] = None,
        config: Optional[AgentConfig] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_name=agent_name,
            services=services,
            config=config or SpreadsheetAgentConfig(),
        )

        # Components
        self.client: Optional[DataFrameClient] = None
        self.llm: Optional[LLMClient] = None
        self.resolver: Optional[SmartDataResolver] = None
        self.state = session_state

        # Streaming support
        self._streaming_sessions: Dict[str, Any] = {}

        # Safe builtins for code execution
        self.SAFE_BUILTINS = {
            "__builtins__": {
                "abs": abs,
                "all": all,
                "any": any,
                "bool": bool,
                "dict": dict,
                "enumerate": enumerate,
                "filter": filter,
                "float": float,
                "int": int,
                "len": len,
                "list": list,
                "map": map,
                "max": max,
                "min": min,
                "range": range,
                "round": round,
                "sorted": sorted,
                "str": str,
                "sum": sum,
                "tuple": tuple,
                "zip": zip,
                "print": print,
                "isinstance": isinstance,
                "type": type,
                "set": set,
                "hasattr": hasattr,
                "getattr": getattr,
                "setattr": setattr,
                "slice": slice,
                "reversed": reversed,
                "iter": iter,
                "next": next,
                "callable": callable,
                "repr": repr,
                "format": format,
                "True": True,
                "False": False,
                "None": None,
            },
            "pd": pd,
            "np": np,
        }

    async def _get_step_context(self, request: AgentRequest, context: ExecutionContext, previous_results: List[Any]) -> Any:
        """Provide detailed context about the active spreadsheets for the ReAct loop."""
        thread_id = request.thread_id or "default"
        session = self.state.get_or_create(thread_id)
        
        if not session.dataframes:
            return "No data loaded yet. Use `load_file` or `upload_file` to load data."
            
        context_str = "CURRENTLY LOADED SPREADSHEETS:\n\n"
        for file_id, df in session.dataframes.items():
            context_str += f"--- File ID: {file_id} ---\n"
            context_str += f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n"
            context_str += f"Columns: {', '.join(df.columns)}\n"
            context_str += f"dtypes:\n{df.dtypes.to_string()}\n\n"
            context_str += f"Data Preview (First 3 rows):\n{df.head(3).to_markdown()}\n\n"
            
        return context_str

    async def _initialize_resources(self):
        """Initialize DataFrame client, LLM client, and resolver."""
        logger.info("Initializing Spreadsheet Agent resources...")

        self.client = DataFrameClient()
        self.llm = LLMClient()
        self.resolver = SmartDataResolver(self.client, self.state)

        # Ensure storage directory exists
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)

        logger.info("Spreadsheet Agent resources initialized")

    async def _cleanup_resources(self):
        """Cleanup resources."""
        logger.info("Cleaning up Spreadsheet Agent resources...")
        cleaned = self.state.cleanup_expired()
        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} expired sessions")

    async def _get_custom_metrics(self) -> Optional[Dict[str, Any]]:
        """Return spreadsheet-specific metrics."""
        metrics = self.state.get_stats()
        metrics["streaming_sessions"] = len(self._streaming_sessions)
        return metrics

    async def _llm_synthesize_response(
        self,
        results: List[Any],
        understanding: Dict[str, Any],
        request: AgentRequest,
    ) -> AgentResponse:
        """Override to add Spreadsheet specific canvas and metadata formatting."""
        
        # 1. Base synthesis using standard BaseAgent fallback if desired or our own
        # Actually, let's just use a simple structured generation for synthesis
        final_answer = ""
        success = True
        
        # Provide text synthesis first
        from pydantic import BaseModel, Field
        class Synthesis(BaseModel):
            summary: str = Field(description="Summary of what was achieved.")
            is_success: bool = Field(description="Did the agent succeed?")
            
        sys_prompt = "You are synthesizing the final outcome of the Spreadsheet Agent execution.\n"
        sys_prompt += "Review the task and the results of the steps to provide a final summary."
        
        user_content = f"Task: {request.prompt}\n\nStep Results:\n"
        for i, res in enumerate(results, 1):
            msg = (res.metadata or {}).get('message', str(res.data)[:200] if res.data else res.error or 'No details')
            user_content += f"Step {i} ({'Success' if res.success else 'Failed'}): {msg}\n"
            
        from langchain_core.messages import SystemMessage, HumanMessage
        synthesis = await self.services.inference.generate_structured(
            messages=[SystemMessage(content=sys_prompt), HumanMessage(content=user_content)],
            schema=Synthesis,
            temperature=0.0
        )
        
        final_answer = synthesis.summary
        success = synthesis.is_success
        
        # 2. Grab the canvas from the last successful step if one was provided
        # or calculate dynamically
        canvas_display = None
        extracted_data = {}
        
        for res in reversed(results):
            if res.success and isinstance(res.data, dict):
                extracted_data.update(res.data)
            # Also check metadata for extra keys (message, canvas_display, etc.)
            if res.success and isinstance(res.metadata, dict):
                extracted_data.update({k: v for k, v in res.metadata.items() if k != 'message'})
                
        # Some steps return explicit canvas displays, check the agent response data?
        # Actually, capabilities returned CapabilityResult which doesn't natively have canvas_display...
        # Let's see if we put canvas_display into CapabilityResult data?
        # No, capability decorator drops things not in `Dict` or extracts it?
        # Wait, inside base agent, we just return CapabilityResult(success=True, data=capability_result, ...)
        # Let's check capabilities
        
        # Capability results are raw returns from our method. We did:
        # return {"success": True, "data": ..., "canvas_display": canvas, ...}
        # If capability decorator wraps it, it might pass through additional fields.
        
        # Let's extract canvas display from results if any
        for res in reversed(results):
            if res.success:
                # Check both data and metadata for canvas_display
                if isinstance(res.data, dict) and "canvas_display" in res.data:
                    canvas_display = res.data["canvas_display"]
                    break
                if isinstance(res.metadata, dict) and "canvas_display" in res.metadata:
                    canvas_display = res.metadata["canvas_display"]
                    break
                
        # 3. Dynamic Canvas Generation Fallback
        if not canvas_display:
            canvas_display = await self._resolve_canvas_dynamic(output=final_answer)

        # 4. Formulate Request
        formatted_data = {
            "task_summary": final_answer,
            "extracted_data": extracted_data,
            "canvas_display": canvas_display
        }
        
        return AgentResponse(
            status="success" if success else "error",
            result=formatted_data,
            error_message=None if success else "Failed to complete some steps",
        )

    async def _update_state_post_step(self, step: Any, result: Any, context: ExecutionContext) -> None:
        """Update session state with step execution outcome."""
        thread_id = context.thread_id or "default"
        session = self.state.get_or_create(thread_id)
        
        # Log action to session history using Session.add_operation
        description = f"Params: {json.dumps(step.parameters, default=str)}\n"
        if result.success:
            msg = (result.metadata or {}).get('message', str(result.data)[:200] if result.data else 'OK')
            description += f"Result: {msg}\n"
        else:
            description += f"Error: {result.error}\n"
            
        session.add_operation(
            action=step.capability_name,
            description=description,
            result_summary=str(result.data)[:300] if result.data else None
        )

    def _extract_prompt(self, params: Dict[str, Any]) -> Optional[str]:
        """Extract prompt from various parameter fields."""
        if not params:
            return None

        fields = ["prompt", "query", "instruction", "q", "content", "message", "action"]
        for field in fields:
            if params.get(field):
                return str(params[field])
        return None

    def _sanitize_for_json(self, obj: Any) -> Any:
        """Sanitize object for JSON serialization."""
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, pd.DataFrame):
            return self._sanitize_for_json(obj.head(50).to_dict(orient="records"))
        if isinstance(obj, pd.Series):
            return self._sanitize_for_json(obj.to_dict())
        if hasattr(obj, "item") and hasattr(obj, "dtype"):
            return obj.item()
        if isinstance(obj, dict):
            return {str(k): self._sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [self._sanitize_for_json(v) for v in obj]
        return str(obj)

    def _build_canvas(
        self, df: pd.DataFrame, title: str, file_id: str = None
    ) -> Dict[str, Any]:
        """Build canvas display for frontend."""
        if not CANVAS_AVAILABLE:
            # Fallback canvas representation
            return {
                "canvas_type": "spreadsheet",
                "canvas_title": title,
                "canvas_data": {
                    "columns": df.columns.tolist(),
                    "data": df.head(100).to_dict("records"),
                    "shape": df.shape,
                    "file_id": file_id,
                },
                "file_id": file_id,
            }

        try:
            display = CanvasService.build_spreadsheet_view(
                filename=file_id or "spreadsheet", dataframe=df, title=title
            )
            canvas_dict = (
                display.model_dump()
                if hasattr(display, "model_dump")
                else display.dict()
            )
            canvas_dict["file_id"] = file_id
            return canvas_dict
        except Exception as e:
            logger.warning(f"Canvas build failed: {e}")
            return {
                "canvas_type": "spreadsheet",
                "canvas_title": title,
                "canvas_data": {
                    "columns": df.columns.tolist(),
                    "data": df.head(100).to_dict("records"),
                    "shape": df.shape,
                },
            }

    async def _resolve_canvas_dynamic(
        self, output: str, capability_name: str = "", primary_canvas: Dict = None
    ) -> Optional[Dict[str, Any]]:
        """
        LLM-driven canvas decision for non-tabular outputs.
        For tabular data, _build_canvas() is still the primary.
        This handles: aggregated summaries → charts, text answers → markdown, etc.
        """
        if not CANVAS_AVAILABLE or not output or len(output) < 30:
            return primary_canvas
        try:
            display = await CanvasService.decide_canvas_llm(
                output=output[:3000],
                agent_name="spreadsheet_agent",
                capability_name=capability_name,
                primary_canvas_type="spreadsheet",
            )
            if display:
                return display.model_dump() if hasattr(display, "model_dump") else display.dict()
        except Exception as e:
            logger.debug(f"LLM dynamic canvas failed: {e}")
        return primary_canvas

    # ========================================================================
    # CAPABILITIES - File Operations
    # ========================================================================

    @capability(
        name="load_file",
        description="Load a spreadsheet file (CSV, Excel) into the session. WARNING: Only use this if the user explicitly asks to load a NEW file. If a file is already loaded, do NOT use this.",
        parameters=[
            ParameterSchema(
                name="file_path",
                type="string",
                description="Path to the file to load",
                required=True,
            ),
            ParameterSchema(
                name="thread_id",
                type="string",
                description="Session thread ID",
                required=False,
                default="default",
            ),
            ParameterSchema(
                name="file_id",
                type="string",
                description="Custom file ID (optional, uses filename if not provided)",
                required=False,
            ),
        ],
    )
    async def load_file(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Load a spreadsheet file."""
        file_path = params.get("file_path")
        thread_id = params.get("thread_id", "default")
        file_id = params.get("file_id") or os.path.basename(file_path)

        if not file_path:
            return {"success": False, "error": "file_path is required"}

        try:
            df, detection_info = await self.client.load_file(file_path=file_path)

            # Store in session
            self.state.store_dataframe(thread_id, file_id, df, file_path)

            # Build canvas for display
            canvas = self._build_canvas(df, f"Loaded: {file_id}", file_id)

            return {
                "success": True,
                "data": {
                    "file_id": file_id,
                    "shape": df.shape,
                    "columns": df.columns.tolist(),
                    "dtypes": {k: str(v) for k, v in df.dtypes.items()},
                    "detection": detection_info,
                },
                "canvas_display": canvas,
                "message": f"Loaded {file_id}: {df.shape[0]} rows × {df.shape[1]} columns",
            }
        except Exception as e:
            logger.error(f"Load file failed: {e}")
            return {"success": False, "error": f"Failed to load file: {str(e)}"}

    @capability(
        name="upload_file",
        description="Upload and load a file from content bytes",
        parameters=[
            ParameterSchema(
                name="content",
                type="string",
                description="Base64 encoded file content",
                required=True,
            ),
            ParameterSchema(
                name="filename",
                type="string",
                description="Original filename",
                required=True,
            ),
            ParameterSchema(
                name="thread_id",
                type="string",
                description="Session thread ID",
                required=False,
                default="default",
            ),
        ],
    )
    async def upload_file(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Upload and load a file from bytes."""
        import base64

        content_b64 = params.get("content")
        filename = params.get("filename")
        thread_id = params.get("thread_id", "default")

        if not content_b64 or not filename:
            return {"success": False, "error": "content and filename are required"}

        try:
            # Decode base64 content
            content = base64.b64decode(content_b64)

            # Save to storage
            file_path = STORAGE_DIR / filename
            with open(file_path, "wb") as f:
                f.write(content)

            # Load the file
            df, detection_info = await self.client.load_file(
                content=content, filename=filename
            )

            file_id = filename
            self.state.store_dataframe(thread_id, file_id, df, str(file_path))

            canvas = self._build_canvas(df, f"Uploaded: {filename}", file_id)

            return {
                "success": True,
                "data": {
                    "file_id": file_id,
                    "file_path": str(file_path),
                    "shape": df.shape,
                    "columns": df.columns.tolist(),
                },
                "canvas_display": canvas,
                "message": f"Uploaded and loaded {filename}: {df.shape[0]} rows",
            }
        except Exception as e:
            logger.error(f"Upload file failed: {e}")
            return {"success": False, "error": f"Upload failed: {str(e)}"}

    @capability(
        name="export_file",
        description="Export data to CSV, Excel, or JSON file",
        parameters=[
            ParameterSchema(
                name="filename",
                type="string",
                description="Output filename (should end with .csv, .xlsx, or .json)",
                required=True,
            ),
            ParameterSchema(
                name="format",
                type="string",
                description="Export format: csv, xlsx, json",
                required=False,
                default="csv",
                enum=["csv", "xlsx", "json"],
            ),
            ParameterSchema(
                name="file_id",
                type="string",
                description="File ID to export (uses latest if not specified)",
                required=False,
            ),
            ParameterSchema(
                name="thread_id",
                type="string",
                description="Session thread ID",
                required=False,
                default="default",
            ),
        ],
    )
    async def export_file(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Export data to file."""
        filename = params.get("filename", "export.csv")
        format = params.get("format", "csv")
        thread_id = params.get("thread_id", "default")
        file_id = params.get("file_id")

        session = self.state.get_or_create(thread_id)
        if not file_id:
            file_id = session.get_latest_file_id()

        # CRITICAL FIX: If specified file_id not found, try the latest file
        # This handles LLM-generated file_ids that don't match UUID-suffixed IDs
        if file_id and file_id not in session.dataframes:
            latest = session.get_latest_file_id()
            if latest and latest in session.dataframes:
                logger.info(f"[FALLBACK] File '{file_id}' not found in export_file, using latest: '{latest}'")
                file_id = latest

        if not file_id or file_id not in session.dataframes:
            return {"success": False, "error": "No data loaded"}

        df = session.dataframes[file_id]

        try:
            # Ensure proper extension
            target_ext = f".{format.lstrip('.')}"
            if not filename.lower().endswith(target_ext.lower()):
                filename += target_ext

            filepath = STORAGE_DIR / filename

            if format.lower() == "csv":
                df.to_csv(filepath, index=False)
            elif format.lower() == "json":
                df.to_json(filepath, orient="records", indent=2)
            else:  # xlsx
                df.to_excel(filepath, index=False)

            # Register in session
            self.state.store_dataframe(thread_id, filename, df, str(filepath))

            return {
                "success": True,
                "data": {
                    "file_id": filename,
                    "file_path": str(filepath),
                    "format": format,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "generated_files": [
                        {
                            "file_name": filename,
                            "file_path": str(filepath),
                            "file_type": "spreadsheet",
                            "file_id": filename,
                        }
                    ],
                },
                "message": f"Exported to {filename}",
            }

        except Exception as e:
            logger.error(f"Export file failed: {e}")
            return {"success": False, "error": f"Export failed: {str(e)}"}

    # ========================================================================
    # CAPABILITIES - Data Processing
    # ========================================================================

    @capability(
        name="process_data",
        description="Process data using natural language instructions (queries, transformations, analysis)",
        parameters=[
            ParameterSchema(
                name="instruction",
                type="string",
                description="Natural language instruction for data processing",
                required=True,
            ),
            ParameterSchema(
                name="file_id",
                type="string",
                description="File ID to process (uses latest if not specified)",
                required=False,
            ),
            ParameterSchema(
                name="thread_id",
                type="string",
                description="Session thread ID",
                required=False,
                default="default",
            ),
        ],
    )
    async def process_data(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Process data using LLM-generated pandas code."""
        instruction = params.get("instruction", "")
        thread_id = params.get("thread_id", "default")
        file_id = params.get("file_id")

        if not instruction:
            return {"success": False, "error": "instruction is required"}

        session = self.state.get_or_create(thread_id)

        if not file_id:
            file_id = session.get_latest_file_id()
        
        # CRITICAL FIX: If specified file_id not found, try the latest file
        if file_id and file_id not in session.dataframes:
            latest = session.get_latest_file_id()
            if latest and latest in session.dataframes:
                logger.info(f"[FALLBACK] File '{file_id}' not found, using latest: '{latest}'")
                file_id = latest

        if not file_id or file_id not in session.dataframes:
            available = list(session.dataframes.keys()) if session.dataframes else []
            logger.error(f"[process_data] No data loaded. Available files: {available}, requested: {params.get('file_id')}")
            return {
                "success": False,
                "error": f"No data loaded. Available files: {available}",
            }


        df = session.dataframes[file_id]

        try:
            # Build context for LLM
            df_context = await self.client.build_context(df, instruction)

            # Generate code
            answer = await self.llm.answer_question(
                instruction, df_context, session.get_recent_history()
            )

            code = answer.get("code")
            if not code:
                return {
                    "success": True,
                    "data": {"answer": answer.get("answer", "No code generated")},
                    "message": "Query processed",
                }

            # Execute code safely
            exec_globals = {
                **self.SAFE_BUILTINS,
                "df": df.copy(),
            }

            exec(code, exec_globals)

            result_data = exec_globals.get("result")
            modified_df = exec_globals.get("df")

            # Check if DataFrame was modified
            df_modified = False
            if isinstance(modified_df, pd.DataFrame) and not modified_df.equals(df):
                df_modified = True
                self.state.store_dataframe(thread_id, file_id, modified_df)
                df = modified_df

            # Format result
            computed_answer = answer.get("answer", "")
            if isinstance(result_data, (int, float)):
                computed_answer = str(result_data)
            elif isinstance(result_data, str):
                computed_answer = result_data
            elif isinstance(result_data, pd.DataFrame):
                computed_answer = f"DataFrame with {len(result_data)} rows"
                result_data = result_data.head(50).to_dict("records")
            elif isinstance(result_data, pd.Series):
                computed_answer = f"Series with {len(result_data)} items"
                result_data = result_data.head(20).to_dict()

            canvas = (
                self._build_canvas(df, "Processed Data", file_id)
                if df_modified
                else None
            )

            return {
                "success": True,
                "data": {
                    "answer": computed_answer,
                    "data": self._sanitize_for_json(result_data),
                    "code": code,
                    "shape": df.shape if isinstance(df, pd.DataFrame) else None,
                    "modified": df_modified,
                },
                "canvas_display": canvas,
                "message": f"Data processed: {computed_answer[:100]}..."
                if len(str(computed_answer)) > 100
                else f"Data processed: {computed_answer}",
            }

        except Exception as e:
            logger.error(f"Process data failed: {e}")
            return {"success": False, "error": f"Processing failed: {str(e)}"}

    @capability(
        name="transform_data",
        description="Apply a custom transformation using LLM-generated pandas code",
        parameters=[
            ParameterSchema(
                name="instruction",
                type="string",
                description="Transformation instruction (e.g., 'Convert all dates to YYYY-MM-DD format')",
                required=True,
            ),
            ParameterSchema(
                name="file_id",
                type="string",
                description="File ID to transform (uses latest if not specified)",
                required=False,
            ),
            ParameterSchema(
                name="thread_id",
                type="string",
                description="Session thread ID",
                required=False,
                default="default",
            ),
        ],
    )
    async def transform_data(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Apply custom transformation via LLM."""
        instruction = params.get("instruction", "")
        thread_id = params.get("thread_id", "default")
        file_id = params.get("file_id")

        if not instruction:
            return {"success": False, "error": "instruction is required"}

        session = self.state.get_or_create(thread_id)
        if not file_id:
            file_id = session.get_latest_file_id()

        # CRITICAL FIX: If specified file_id not found, try the latest file
        if file_id and file_id not in session.dataframes:
            latest = session.get_latest_file_id()
            if latest and latest in session.dataframes:
                logger.info(f"[FALLBACK] File '{file_id}' not found in transform_data, using latest: '{latest}'")
                file_id = latest

        if not file_id or file_id not in session.dataframes:
            return {"success": False, "error": "No data loaded"}

        df = session.dataframes[file_id]

        try:
            # Generate transformation code
            code = await self.llm.generate_pandas_code(
                instruction, await self.client.build_context(df)
            )

            # Execute
            exec_globals = {
                **self.SAFE_BUILTINS,
                "df": df.copy(),
            }
            exec(code, exec_globals)
            result_df = exec_globals["df"]

            # Store result
            self.state.store_dataframe(thread_id, file_id, result_df)

            canvas = self._build_canvas(result_df, "Transformed Data", file_id)

            return {
                "success": True,
                "data": {
                    "code": code,
                    "shape": result_df.shape,
                    "columns": result_df.columns.tolist(),
                },
                "canvas_display": canvas,
                "message": f"Data transformed: {result_df.shape[0]} rows × {result_df.shape[1]} columns",
            }

        except Exception as e:
            logger.error(f"Transform data failed: {e}")
            return {"success": False, "error": f"Transformation failed: {str(e)}"}

    # ========================================================================
    # CAPABILITIES - Filter & Sort
    # ========================================================================

    @capability(
        name="filter_data",
        description="Filter rows based on column conditions",
        parameters=[
            ParameterSchema(
                name="column",
                type="string",
                description="Column name to filter on",
                required=True,
            ),
            ParameterSchema(
                name="operator",
                type="string",
                description="Comparison operator: ==, !=, >, <, >=, <=, contains, startswith, endswith",
                required=True,
                enum=[
                    "==",
                    "!=",
                    ">",
                    "<",
                    ">=",
                    "<=",
                    "in",
                    "not_in",
                    "contains",
                    "startswith",
                    "endswith",
                ],
            ),
            ParameterSchema(
                name="value",
                type="string",
                description="Value to compare against",
                required=True,
            ),
            ParameterSchema(
                name="file_id",
                type="string",
                description="File ID to filter (uses latest if not specified)",
                required=False,
            ),
            ParameterSchema(
                name="thread_id",
                type="string",
                description="Session thread ID",
                required=False,
                default="default",
            ),
        ],
    )
    async def filter_data(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Filter data based on conditions."""
        column = params.get("column")
        operator = params.get("operator", "==")
        value = params.get("value")
        thread_id = params.get("thread_id", "default")
        file_id = params.get("file_id")

        session = self.state.get_or_create(thread_id)
        if not file_id:
            file_id = session.get_latest_file_id()

        # CRITICAL FIX: If specified file_id not found, try the latest file
        if file_id and file_id not in session.dataframes:
            latest = session.get_latest_file_id()
            if latest and latest in session.dataframes:
                logger.info(f"[FALLBACK] File '{file_id}' not found in filter_data, using latest: '{latest}'")
                file_id = latest

        if not file_id or file_id not in session.dataframes:
            return {"success": False, "error": "No data loaded"}

        df = session.dataframes[file_id]

        try:
            # Resolve column name
            cols = self.resolver.resolve_columns(df, [column])
            if not cols:
                return {"success": False, "error": f"Column not found: {column}"}

            col = cols[0]

            # Apply filter based on operator
            if operator == "==":
                filtered = df[df[col] == value]
            elif operator == "!=":
                filtered = df[df[col] != value]
            elif operator == ">":
                filtered = df[df[col] > float(value)]
            elif operator == "<":
                filtered = df[df[col] < float(value)]
            elif operator == ">=":
                filtered = df[df[col] >= float(value)]
            elif operator == "<=":
                filtered = df[df[col] <= float(value)]
            elif operator == "contains":
                filtered = df[
                    df[col].astype(str).str.contains(str(value), case=False, na=False)
                ]
            elif operator == "in":
                # Support comma-separated or list values for multi-value matching
                if isinstance(value, list):
                    match_values = [str(v).strip() for v in value]
                else:
                    match_values = [v.strip() for v in str(value).split(",")]
                filtered = df[df[col].astype(str).isin(match_values)]
            elif operator == "not_in":
                if isinstance(value, list):
                    match_values = [str(v).strip() for v in value]
                else:
                    match_values = [v.strip() for v in str(value).split(",")]
                filtered = df[~df[col].astype(str).isin(match_values)]
            elif operator == "startswith":
                filtered = df[df[col].astype(str).str.startswith(str(value), na=False)]
            elif operator == "endswith":
                filtered = df[df[col].astype(str).str.endswith(str(value), na=False)]
            else:
                filtered = df[df[col] == value]

            # Store result as new file
            new_file_id = f"{file_id}_filtered_{uuid.uuid4().hex[:8]}"
            self.state.store_dataframe(thread_id, new_file_id, filtered)

            canvas = self._build_canvas(filtered, "Filtered Data", new_file_id)

            return {
                "success": True,
                "data": {
                    "file_id": new_file_id,
                    "original_rows": len(df),
                    "filtered_rows": len(filtered),
                    "removed": len(df) - len(filtered),
                },
                "canvas_display": canvas,
                "message": f"Filtered {len(filtered)} rows from {len(df)}",
            }

        except Exception as e:
            logger.error(f"Filter data failed: {e}")
            return {"success": False, "error": f"Filter failed: {str(e)}"}

    @capability(
        name="sort_data",
        description="Sort data by one or more columns",
        parameters=[
            ParameterSchema(
                name="columns",
                type="array",
                description="List of column names to sort by",
                required=True,
            ),
            ParameterSchema(
                name="ascending",
                type="boolean",
                description="Sort in ascending order (default: true)",
                required=False,
                default=True,
            ),
            ParameterSchema(
                name="file_id",
                type="string",
                description="File ID to sort (uses latest if not specified)",
                required=False,
            ),
            ParameterSchema(
                name="thread_id",
                type="string",
                description="Session thread ID",
                required=False,
                default="default",
            ),
        ],
    )
    async def sort_data(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Sort data by columns."""
        columns = params.get("columns", [])
        ascending = params.get("ascending", True)
        thread_id = params.get("thread_id", "default")
        file_id = params.get("file_id")

        if isinstance(columns, str):
            columns = [columns]

        session = self.state.get_or_create(thread_id)
        if not file_id:
            file_id = session.get_latest_file_id()

        # CRITICAL FIX: If specified file_id not found, try the latest file
        if file_id and file_id not in session.dataframes:
            latest = session.get_latest_file_id()
            if latest and latest in session.dataframes:
                logger.info(f"[FALLBACK] File '{file_id}' not found in sort_data, using latest: '{latest}'")
                file_id = latest

        if not file_id or file_id not in session.dataframes:
            return {"success": False, "error": "No data loaded"}

        df = session.dataframes[file_id]

        try:
            # Resolve column names
            resolved_cols = self.resolver.resolve_columns(df, columns)

            if not resolved_cols:
                return {"success": False, "error": f"Columns not found: {columns}"}

            sorted_df = df.sort_values(by=resolved_cols, ascending=ascending)

            # Store result
            new_file_id = f"{file_id}_sorted_{uuid.uuid4().hex[:8]}"
            self.state.store_dataframe(thread_id, new_file_id, sorted_df)

            canvas = self._build_canvas(sorted_df, "Sorted Data", new_file_id)

            return {
                "success": True,
                "data": {
                    "file_id": new_file_id,
                    "sorted_by": resolved_cols,
                    "ascending": ascending,
                },
                "canvas_display": canvas,
                "message": f"Sorted by {', '.join(resolved_cols)}",
            }

        except Exception as e:
            logger.error(f"Sort data failed: {e}")
            return {"success": False, "error": f"Sort failed: {str(e)}"}

    # ========================================================================
    # CAPABILITIES - Aggregate & Merge
    # ========================================================================

    @capability(
        name="aggregate_data",
        description="Group and aggregate data (sum, mean, count, min, max, nunique)",
        parameters=[
            ParameterSchema(
                name="group_by",
                type="string",
                description="Column to group by",
                required=True,
            ),
            ParameterSchema(
                name="agg_column",
                type="string",
                description="Column to aggregate",
                required=True,
            ),
            ParameterSchema(
                name="function",
                type="string",
                description="Aggregation function: sum, mean, count, min, max, nunique, std, var, median",
                required=True,
                enum=[
                    "sum",
                    "mean",
                    "count",
                    "min",
                    "max",
                    "nunique",
                    "std",
                    "var",
                    "median",
                ],
            ),
            ParameterSchema(
                name="file_id",
                type="string",
                description="File ID to aggregate (uses latest if not specified)",
                required=False,
            ),
            ParameterSchema(
                name="thread_id",
                type="string",
                description="Session thread ID",
                required=False,
                default="default",
            ),
        ],
    )
    async def aggregate_data(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Aggregate data by groups."""
        group_by = params.get("group_by")
        agg_column = params.get("agg_column")
        function = params.get("function", "sum")
        thread_id = params.get("thread_id", "default")
        file_id = params.get("file_id")

        session = self.state.get_or_create(thread_id)
        if not file_id:
            file_id = session.get_latest_file_id()
        
        # CRITICAL FIX: If specified file_id not found, try the latest file
        # This handles LLM placeholder file_ids like "sales_data_aggregated_mean"
        if file_id and file_id not in session.dataframes:
            latest = session.get_latest_file_id()
            if latest and latest in session.dataframes:
                logger.info(f"[FALLBACK] File '{file_id}' not found, using latest: '{latest}'")
                file_id = latest

        if not file_id or file_id not in session.dataframes:
            available = list(session.dataframes.keys()) if session.dataframes else []
            logger.error(f"[aggregate_data] No data loaded. Available files: {available}, requested: {params.get('file_id')}")
            return {"success": False, "error": f"No data loaded. Available files: {available}"}


        df = session.dataframes[file_id]

        try:
            # Resolve columns
            group_cols = self.resolver.resolve_columns(df, [group_by])
            agg_cols = self.resolver.resolve_columns(df, [agg_column])

            if not group_cols:
                return {
                    "success": False,
                    "error": f"Group column not found: {group_by}",
                }
            if not agg_cols:
                return {
                    "success": False,
                    "error": f"Aggregate column not found: {agg_column}",
                }

            group_col = group_cols[0]
            agg_col = agg_cols[0]

            # Map function name to pandas aggregation
            agg_funcs = {
                "sum": "sum",
                "mean": "mean",
                "count": "count",
                "min": "min",
                "max": "max",
                "nunique": "nunique",
                "std": "std",
                "var": "var",
                "median": "median",
            }

            agg_func = agg_funcs.get(function, "sum")
            result = df.groupby(group_col)[agg_col].agg(agg_func).reset_index()

            # Store result
            result_file_id = f"{file_id}_aggregated_{uuid.uuid4().hex[:8]}"
            self.state.store_dataframe(thread_id, result_file_id, result)

            canvas = self._build_canvas(result, "Aggregated Data", result_file_id)

            return {
                "success": True,
                "data": {
                    "file_id": result_file_id,
                    "groups": len(result),
                    "aggregation": function,
                    "data": result.head(100).to_dict("records"),
                },
                "canvas_display": canvas,
                "message": f"Aggregated {len(result)} groups by {function}",
            }

        except Exception as e:
            logger.error(f"Aggregate data failed: {e}")
            return {"success": False, "error": f"Aggregation failed: {str(e)}"}

    @capability(
        name="merge_data",
        description="Merge/join two or more dataframes",
        parameters=[
            ParameterSchema(
                name="file_ids",
                type="array",
                description="List of file IDs to merge (at least 2 required)",
                required=True,
            ),
            ParameterSchema(
                name="how",
                type="string",
                description="Type of merge: inner, left, right, outer",
                required=False,
                default="inner",
                enum=["inner", "left", "right", "outer"],
            ),
            ParameterSchema(
                name="on",
                type="string",
                description="Column(s) to merge on (optional, will infer if not provided)",
                required=False,
            ),
            ParameterSchema(
                name="thread_id",
                type="string",
                description="Session thread ID",
                required=False,
                default="default",
            ),
        ],
    )
    async def merge_data(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Merge/join dataframes."""
        file_ids = params.get("file_ids", [])
        how = params.get("how", "inner")
        on = params.get("on")
        thread_id = params.get("thread_id", "default")

        if len(file_ids) < 2:
            return {
                "success": False,
                "error": "At least 2 file_ids are required for merge",
            }

        session = self.state.get_or_create(thread_id)

        try:
            # Resolve dataframes
            dfs_to_merge = []
            for fid in file_ids:
                if fid in session.dataframes:
                    dfs_to_merge.append(session.dataframes[fid])
                else:
                    return {"success": False, "error": f"File ID not found: {fid}"}

            # Perform merge (first 2 dataframes)
            left = dfs_to_merge[0]
            right = dfs_to_merge[1]

            merge_kwargs = {"how": how}
            if on:
                cols = self.resolver.resolve_columns(left, [on])
                if cols:
                    merge_kwargs["on"] = cols[0]

            result_df = pd.merge(left, right, **merge_kwargs)

            # If more than 2, continue merging
            for i in range(2, len(dfs_to_merge)):
                result_df = pd.merge(result_df, dfs_to_merge[i], **merge_kwargs)

            # Store result
            result_file_id = f"merged_{uuid.uuid4().hex[:8]}"
            self.state.store_dataframe(thread_id, result_file_id, result_df)

            canvas = self._build_canvas(result_df, "Merged Data", result_file_id)

            return {
                "success": True,
                "data": {
                    "file_id": result_file_id,
                    "rows": len(result_df),
                    "columns": len(result_df.columns),
                    "source_files": file_ids,
                },
                "canvas_display": canvas,
                "message": f"Merged {len(file_ids)} files into {len(result_df)} rows",
            }

        except Exception as e:
            logger.error(f"Merge data failed: {e}")
            return {"success": False, "error": f"Merge failed: {str(e)}"}

    # ========================================================================
    # CAPABILITIES - Column Management
    # ========================================================================

    @capability(
        name="add_column",
        description="Add a new calculated column",
        parameters=[
            ParameterSchema(
                name="name",
                type="string",
                description="Name for the new column",
                required=True,
            ),
            ParameterSchema(
                name="expression",
                type="string",
                description="Expression or instruction for the column value (e.g., df['Price'] * df['Quantity'] or 'Calculate total price')",
                required=True,
            ),
            ParameterSchema(
                name="file_id",
                type="string",
                description="File ID (uses latest if not specified)",
                required=False,
            ),
            ParameterSchema(
                name="thread_id",
                type="string",
                description="Session thread ID",
                required=False,
                default="default",
            ),
        ],
    )
    async def add_column(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Add a calculated column."""
        name = params.get("name")
        expression = params.get("expression")
        thread_id = params.get("thread_id", "default")
        file_id = params.get("file_id")

        if not name or not expression:
            return {"success": False, "error": "name and expression are required"}

        session = self.state.get_or_create(thread_id)
        if not file_id:
            file_id = session.get_latest_file_id()

        # CRITICAL FIX: If specified file_id not found, try the latest file
        if file_id and file_id not in session.dataframes:
            latest = session.get_latest_file_id()
            if latest and latest in session.dataframes:
                logger.info(f"[FALLBACK] File '{file_id}' not found in add_column, using latest: '{latest}'")
                file_id = latest

        if not file_id or file_id not in session.dataframes:
            return {"success": False, "error": "No data loaded"}

        df = session.dataframes[file_id]

        try:
            # Generate code if expression is natural language
            if not expression.strip().startswith("df["):
                code = await self.llm.generate_pandas_code(
                    f"Add a new column called '{name}' where: {expression}",
                    await self.client.build_context(df),
                )
            else:
                code = f"df['{name}'] = {expression}"

            # Execute
            exec_globals = {
                **self.SAFE_BUILTINS,
                "df": df.copy(),
            }
            exec(code, exec_globals)
            result_df = exec_globals["df"]

            # Store result
            self.state.store_dataframe(thread_id, file_id, result_df)

            canvas = self._build_canvas(result_df, f"Added Column: {name}", file_id)

            return {
                "success": True,
                "data": {"column": name, "code": code, "shape": result_df.shape},
                "canvas_display": canvas,
                "message": f"Added column '{name}'",
            }

        except Exception as e:
            logger.error(f"Add column failed: {e}")
            return {"success": False, "error": f"Failed to add column: {str(e)}"}

    @capability(
        name="drop_column",
        description="Drop/remove a column",
        parameters=[
            ParameterSchema(
                name="column",
                type="string",
                description="Column name to drop",
                required=True,
            ),
            ParameterSchema(
                name="file_id",
                type="string",
                description="File ID (uses latest if not specified)",
                required=False,
            ),
            ParameterSchema(
                name="thread_id",
                type="string",
                description="Session thread ID",
                required=False,
                default="default",
            ),
        ],
    )
    async def drop_column(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Drop a column."""
        column = params.get("column")
        thread_id = params.get("thread_id", "default")
        file_id = params.get("file_id")

        if not column:
            return {"success": False, "error": "column is required"}

        session = self.state.get_or_create(thread_id)
        if not file_id:
            file_id = session.get_latest_file_id()

        # CRITICAL FIX: If specified file_id not found, try the latest file
        if file_id and file_id not in session.dataframes:
            latest = session.get_latest_file_id()
            if latest and latest in session.dataframes:
                logger.info(f"[FALLBACK] File '{file_id}' not found in drop_column, using latest: '{latest}'")
                file_id = latest

        if not file_id or file_id not in session.dataframes:
            return {"success": False, "error": "No data loaded"}

        df = session.dataframes[file_id]

        try:
            cols = self.resolver.resolve_columns(df, [column])
            if not cols:
                return {"success": False, "error": f"Column not found: {column}"}

            result_df = df.drop(columns=cols)

            # Store result
            self.state.store_dataframe(thread_id, file_id, result_df)

            canvas = self._build_canvas(
                result_df, f"Dropped Column: {cols[0]}", file_id
            )

            return {
                "success": True,
                "data": {
                    "dropped": cols[0],
                    "remaining_columns": len(result_df.columns),
                },
                "canvas_display": canvas,
                "message": f"Dropped column '{cols[0]}'",
            }

        except Exception as e:
            logger.error(f"Drop column failed: {e}")
            return {"success": False, "error": f"Failed to drop column: {str(e)}"}

    @capability(
        name="rename_column",
        description="Rename a column",
        parameters=[
            ParameterSchema(
                name="old_name",
                type="string",
                description="Current column name",
                required=True,
            ),
            ParameterSchema(
                name="new_name",
                type="string",
                description="New column name",
                required=True,
            ),
            ParameterSchema(
                name="file_id",
                type="string",
                description="File ID (uses latest if not specified)",
                required=False,
            ),
            ParameterSchema(
                name="thread_id",
                type="string",
                description="Session thread ID",
                required=False,
                default="default",
            ),
        ],
    )
    async def rename_column(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Rename a column."""
        old_name = params.get("old_name")
        new_name = params.get("new_name")
        thread_id = params.get("thread_id", "default")
        file_id = params.get("file_id")

        if not old_name or not new_name:
            return {"success": False, "error": "old_name and new_name are required"}

        session = self.state.get_or_create(thread_id)
        if not file_id:
            file_id = session.get_latest_file_id()

        # CRITICAL FIX: If specified file_id not found, try the latest file
        if file_id and file_id not in session.dataframes:
            latest = session.get_latest_file_id()
            if latest and latest in session.dataframes:
                logger.info(f"[FALLBACK] File '{file_id}' not found in rename_column, using latest: '{latest}'")
                file_id = latest

        if not file_id or file_id not in session.dataframes:
            return {"success": False, "error": "No data loaded"}

        df = session.dataframes[file_id]

        try:
            cols = self.resolver.resolve_columns(df, [old_name])
            if not cols:
                return {"success": False, "error": f"Column not found: {old_name}"}

            result_df = df.rename(columns={cols[0]: new_name})

            # Store result
            self.state.store_dataframe(thread_id, file_id, result_df)

            canvas = self._build_canvas(
                result_df, f"Renamed: {cols[0]} → {new_name}", file_id
            )

            return {
                "success": True,
                "data": {"old": cols[0], "new": new_name},
                "canvas_display": canvas,
                "message": f"Renamed column '{cols[0]}' to '{new_name}'",
            }

        except Exception as e:
            logger.error(f"Rename column failed: {e}")
            return {"success": False, "error": f"Failed to rename column: {str(e)}"}

    @capability(
        name="fill_missing",
        description="Fill missing/NaN values in a column",
        parameters=[
            ParameterSchema(
                name="column",
                type="string",
                description="Column to fill (if not provided, fills all columns)",
                required=False,
            ),
            ParameterSchema(
                name="value",
                type="string",
                description="Value to fill with (default: 0)",
                required=False,
                default="0",
            ),
            ParameterSchema(
                name="method",
                type="string",
                description="Fill method: value, ffill (forward fill), bfill (backward fill), mean, median",
                required=False,
                default="value",
                enum=["value", "ffill", "bfill", "mean", "median"],
            ),
            ParameterSchema(
                name="file_id",
                type="string",
                description="File ID (uses latest if not specified)",
                required=False,
            ),
            ParameterSchema(
                name="thread_id",
                type="string",
                description="Session thread ID",
                required=False,
                default="default",
            ),
        ],
    )
    async def fill_missing(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Fill missing values."""
        column = params.get("column")
        value = params.get("value", "0")
        method = params.get("method", "value")
        thread_id = params.get("thread_id", "default")
        file_id = params.get("file_id")

        session = self.state.get_or_create(thread_id)
        if not file_id:
            file_id = session.get_latest_file_id()

        # CRITICAL FIX: If specified file_id not found, try the latest file
        if file_id and file_id not in session.dataframes:
            latest = session.get_latest_file_id()
            if latest and latest in session.dataframes:
                logger.info(f"[FALLBACK] File '{file_id}' not found in fill_missing, using latest: '{latest}'")
                file_id = latest

        if not file_id or file_id not in session.dataframes:
            return {"success": False, "error": "No data loaded"}

        df = session.dataframes[file_id]

        try:
            result_df = df.copy()

            if column:
                cols = self.resolver.resolve_columns(df, [column])
                if not cols:
                    return {"success": False, "error": f"Column not found: {column}"}

                col = cols[0]

                if method == "value":
                    result_df[col] = result_df[col].fillna(value)
                elif method == "ffill":
                    result_df[col] = result_df[col].fillna(method="ffill")
                elif method == "bfill":
                    result_df[col] = result_df[col].fillna(method="bfill")
                elif method == "mean":
                    result_df[col] = result_df[col].fillna(result_df[col].mean())
                elif method == "median":
                    result_df[col] = result_df[col].fillna(result_df[col].median())
            else:
                # Fill all columns
                if method == "value":
                    result_df = result_df.fillna(value)
                elif method == "ffill":
                    result_df = result_df.fillna(method="ffill")
                elif method == "bfill":
                    result_df = result_df.fillna(method="bfill")

            # Store result
            self.state.store_dataframe(thread_id, file_id, result_df)

            # Count filled values
            filled_count = df.isna().sum().sum() - result_df.isna().sum().sum()

            canvas = self._build_canvas(result_df, "Filled Missing Values", file_id)

            return {
                "success": True,
                "data": {
                    "filled_count": int(filled_count),
                    "method": method,
                    "column": column or "all",
                },
                "canvas_display": canvas,
                "message": f"Filled {filled_count} missing values",
            }

        except Exception as e:
            logger.error(f"Fill missing failed: {e}")
            return {
                "success": False,
                "error": f"Failed to fill missing values: {str(e)}",
            }

    # ========================================================================
    # CAPABILITIES - Info & Summary
    # ========================================================================

    @capability(
        name="get_summary",
        description="Get a comprehensive summary of the data including statistics and column info",
        parameters=[
            ParameterSchema(
                name="file_id",
                type="string",
                description="File ID to summarize (uses latest if not specified)",
                required=False,
            ),
            ParameterSchema(
                name="thread_id",
                type="string",
                description="Session thread ID",
                required=False,
                default="default",
            ),
        ],
    )
    async def get_summary(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Get data summary."""
        thread_id = params.get("thread_id", "default")
        file_id = params.get("file_id")

        session = self.state.get_or_create(thread_id)
        if not file_id:
            file_id = session.get_latest_file_id()

        # CRITICAL FIX: If specified file_id not found, try the latest file
        if file_id and file_id not in session.dataframes:
            latest = session.get_latest_file_id()
            if latest and latest in session.dataframes:
                logger.info(f"[FALLBACK] File '{file_id}' not found in get_summary, using latest: '{latest}'")
                file_id = latest

        if not file_id or file_id not in session.dataframes:
            return {"success": False, "error": "No data loaded"}

        df = session.dataframes[file_id]

        try:
            # Get statistics
            summary = {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": {k: str(v) for k, v in df.dtypes.items()},
                "memory_usage_mb": round(
                    df.memory_usage(deep=True).sum() / (1024 * 1024), 2
                ),
                "null_counts": df.isnull().sum().to_dict(),
                "null_percentage": {
                    k: round(v / len(df) * 100, 2)
                    for k, v in df.isnull().sum().to_dict().items()
                    if v > 0
                },
            }

            # Numeric column statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                summary["numeric_stats"] = df[numeric_cols].describe().to_dict()

            # Categorical columns
            categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
            if categorical_cols:
                summary["categorical_counts"] = {
                    col: df[col].nunique() for col in categorical_cols[:5]
                }

            # Sample data (first 5 rows)
            summary["sample"] = df.head(5).to_dict("records")

            canvas = self._build_canvas(df.head(100), f"Summary: {file_id}", file_id)

            return {
                "success": True,
                "data": summary,
                "canvas_display": canvas,
                "message": f"Summary: {df.shape[0]} rows × {df.shape[1]} columns",
            }

        except Exception as e:
            logger.error(f"Get summary failed: {e}")
            return {"success": False, "error": f"Summary failed: {str(e)}"}

    @capability(
        name="list_files",
        description="List all files in the current session",
        parameters=[
            ParameterSchema(
                name="thread_id",
                type="string",
                description="Session thread ID",
                required=False,
                default="default",
            )
        ],
    )
    async def list_files(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """List all files in session."""
        thread_id = params.get("thread_id", "default")

        session = self.state.get(thread_id)
        if not session:
            return {
                "success": True,
                "data": {"files": [], "count": 0},
                "message": "No files in session",
            }

        files = []
        for file_id, metadata in session.file_metadata.items():
            files.append(
                {
                    "file_id": file_id,
                    "file_path": session.file_paths.get(file_id, ""),
                    "rows": metadata.get("rows", 0),
                    "columns": metadata.get("columns", 0),
                    "column_names": metadata.get("column_names", []),
                }
            )

        return {
            "success": True,
            "data": {"files": files, "count": len(files)},
            "message": f"Found {len(files)} files in session",
        }
