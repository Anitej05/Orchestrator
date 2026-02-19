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
        if file_id and file_id not in session.dataframes:
            latest = session.get_latest_file_id()
            if latest and latest in session.dataframes:
                logger.info(f"[FALLBACK] File '{file_id}' not found, using latest: '{latest}'")
                file_id = latest

        if not file_id or file_id not in session.dataframes:
            available = list(session.dataframes.keys()) if session.dataframes else []
            logger.error(f"[export_file] No data loaded. Available files: {available}, requested: {params.get('file_id')}")
            return {"success": False, "error": f"No data loaded. Available files: {available}"}

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
                logger.info(f"[FALLBACK] File '{file_id}' not found, using latest: '{latest}'")
                file_id = latest

        if not file_id or file_id not in session.dataframes:
            available = list(session.dataframes.keys()) if session.dataframes else []
            logger.error(f"[transform_data] No data loaded. Available files: {available}, requested: {params.get('file_id')}")
            return {"success": False, "error": f"No data loaded. Available files: {available}"}

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
                logger.info(f"[FALLBACK] File '{file_id}' not found, using latest: '{latest}'")
                file_id = latest

        if not file_id or file_id not in session.dataframes:
            available = list(session.dataframes.keys()) if session.dataframes else []
            logger.error(f"[filter_data] No data loaded. Available files: {available}, requested: {params.get('file_id')}")
            return {"success": False, "error": f"No data loaded. Available files: {available}"}

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
                logger.info(f"[FALLBACK] File '{file_id}' not found, using latest: '{latest}'")
                file_id = latest

        if not file_id or file_id not in session.dataframes:
            available = list(session.dataframes.keys()) if session.dataframes else []
            logger.error(f"[sort_data] No data loaded. Available files: {available}, requested: {params.get('file_id')}")
            return {"success": False, "error": f"No data loaded. Available files: {available}"}

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
