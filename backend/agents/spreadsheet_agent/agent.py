"""
Spreadsheet Agent - Simplified Main Agent

Clean, focused implementation with:
- Unified /execute endpoint
- LLM-powered task decomposition
- Simple file handling
- Clean error handling
"""

from __future__ import annotations  # Enable forward references for type hints

import logging
import os
import re
from typing import Dict, Any, Optional, List
from pathlib import Path

import pandas as pd

from .config import logger
from .agent_schemas import ExecuteResponse, StepResult, TaskStatus, StepPlan
from .state import session_state, Session
from .client import df_client, SmartDataResolver
from .llm_helpers import SpreadsheetLLMHelpers

from backend.services.content_management_service import ContentManagementService
from backend.services.canvas_service import CanvasService
from backend.schemas import AgentResponse, StandardAgentResponse, AgentResponseStatus

logger = logging.getLogger("spreadsheet_agent.agent")


class SpreadsheetAgent(SpreadsheetLLMHelpers):
    """
    Central orchestrator for spreadsheet operations.

    Inherits LLM methods from SpreadsheetLLMHelpers:
    - decompose_request(), generate_pandas_code(), answer_question(), etc.
    """

    # Safe builtins for code execution (class-level constant)
    SAFE_BUILTINS: Dict[str, Any] = {
        'abs': abs, 'all': all, 'any': any, 'bool': bool, 'dict': dict,
        'enumerate': enumerate, 'float': float, 'int': int, 'len': len,
        'list': list, 'max': max, 'min': min, 'range': range, 'set': set,
        'str': str, 'sum': sum, 'tuple': tuple, 'zip': zip
    }

    def __init__(self) -> None:
        self.state = session_state
        self.client = df_client
        self.resolver = SmartDataResolver(self.client, self.state)
        self.cms = ContentManagementService()
        logger.info("SpreadsheetAgent initialized")

    def _extract_prompt(self, params: Dict[str, Any]) -> Optional[str]:
        """Extract prompt from parameters."""
        if not params:
            return None
        for field in ['prompt', 'query', 'instruction', 'q', 'content', 'message']:
            value = params.get(field)
            if value:
                return str(value)
        return None

    async def execute(
        self,
        prompt: Optional[str] = None,
        action: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        thread_id: str = "default",
        file_content: Optional[bytes] = None,
        filename: Optional[str] = None
    ) -> AgentResponse:
        """Unified execution endpoint."""
        params = params or {}

        try:
            session = self.state.get_or_create(thread_id)

            # Extract prompt if not provided
            if not prompt:
                prompt = self._extract_prompt(params) or ""

            # Handle file upload
            if file_content and filename:
                upload_result = await self._handle_file_upload(file_content, filename, thread_id, session)
                if not prompt:
                    return self._to_agent_response(upload_result)
                if not upload_result.success:
                    return self._to_agent_response(upload_result)

            # Auto-load file from path if provided
            file_path = params.get('file_path')
            if not file_path and prompt:
                file_path = self._extract_file_path(prompt)

            if file_path and not session.dataframes:
                try:
                    df, _ = await self.client.load_file(file_path=file_path)
                    file_id = os.path.basename(file_path)
                    self.state.store_dataframe(thread_id, file_id, df, file_path)
                except Exception as e:
                    return self._to_agent_response(ExecuteResponse(
                        status=TaskStatus.ERROR,
                        success=False,
                        error=f"Failed to load file: {e}"
                    ))

            # Execute based on input type
            if prompt and not action:
                exec_res = await self._execute_complex(prompt, thread_id, session, params)
            elif action:
                exec_res = await self._execute_action(action, params, thread_id, session)
            elif session.get_latest_file_id():
                prompt = "Provide a comprehensive summary of this data."
                exec_res = await self._execute_complex(prompt, thread_id, session, params)
            else:
                exec_res = ExecuteResponse(
                    status=TaskStatus.ERROR,
                    success=False,
                    error="Either 'prompt' or 'action' must be provided"
                )

            return self._to_agent_response(exec_res)

        except Exception as e:
            logger.error(f"Execute failed: {e}")
            return self._to_agent_response(ExecuteResponse(
                status=TaskStatus.ERROR,
                success=False,
                error=str(e)
            ))

    def _extract_file_path(self, prompt: str) -> Optional[str]:
        """Extract file path from prompt text."""
        # Match Windows paths like D:\path\to\file.xlsx
        match = re.search(r'([A-Za-z]:[\\/][^\s,;\'"]+\.(?:csv|xlsx?|xls))', prompt, re.IGNORECASE)
        if match:
            path = match.group(1).replace('/', os.sep).replace('\\', os.sep)
            if os.path.exists(path):
                return path
        return None

    def _to_agent_response(self, exec_res: ExecuteResponse) -> AgentResponse:
        """Convert ExecuteResponse to AgentResponse."""
        # Handle canvas_display - it can be a dict or CanvasDisplay object
        canvas_display_dict = None
        if exec_res.canvas_display:
            if hasattr(exec_res.canvas_display, 'model_dump'):
                canvas_display_dict = exec_res.canvas_display.model_dump()
            elif isinstance(exec_res.canvas_display, dict):
                canvas_display_dict = exec_res.canvas_display

        std_response = StandardAgentResponse(
            status="success" if exec_res.success else "error",
            summary=exec_res.message or ("Success" if exec_res.success else "Failed"),
            data=exec_res.result or {},
            canvas_display=canvas_display_dict,
            error_message=exec_res.error
        )
        return AgentResponse(
            status=AgentResponseStatus.COMPLETE if exec_res.success else AgentResponseStatus.ERROR,
            result=exec_res.model_dump() if hasattr(exec_res, 'model_dump') else exec_res.dict(),
            standard_response=std_response,
            error=exec_res.error
        )

    async def _handle_file_upload(
        self,
        file_content: bytes,
        filename: str,
        thread_id: str,
        session: Session
    ) -> ExecuteResponse:
        """Handle file upload from content."""
        try:
            file_id = session.save_file(file_content, filename)
            df, detection = await self.client.load_file(file_id=file_id)
            self.state.store_dataframe(thread_id, filename, df, str(file_id))

            return ExecuteResponse(
                status=TaskStatus.COMPLETED,
                success=True,
                message=f"Loaded {filename} ({df.shape[0]} rows, {df.shape[1]} columns)",
                data={"file_id": str(file_id), "shape": df.shape}
            )
        except Exception as e:
            return ExecuteResponse(
                status=TaskStatus.ERROR,
                success=False,
                error=str(e)
            )

    async def _execute_complex(
        self,
        prompt: str,
        thread_id: str,
        session: Session,
        params: Dict[str, Any],
        max_retries: int = 2
    ) -> ExecuteResponse:
        """LLM-powered task decomposition and execution."""
        logger.info(f"[COMPLEX] Processing: {prompt[:100]}...")

        execution_errors: List[Dict[str, str]] = []

        for attempt in range(max_retries):
            context = await self._build_context(session, prompt)
            error_summary: Optional[str] = "\n".join([f"- {e['step']}: {e['error']}" for e in execution_errors[-3:]]) if execution_errors else None

            plan = await self.decompose_request(prompt, context, error_context=error_summary)

            if plan.needs_clarification:
                task_id = f"task-{len(session.pending_tasks)}"
                self.state.pause_task(thread_id, task_id, plan.question or "", {
                    "original_prompt": prompt,
                    "plan": plan.model_dump() if hasattr(plan, 'model_dump') else plan.dict()
                })
                return ExecuteResponse(
                    status=TaskStatus.NEEDS_INPUT,
                    success=True,
                    question=plan.question,
                    question_type="choice" if plan.options else "text",
                    options=plan.options
                )

            results: List[StepResult] = []
            for i, step in enumerate(plan.steps):
                result = await self._execute_step(step, session, thread_id)
                results.append(result)

                if not result.success:
                    execution_errors.append({
                        'step': step.action,
                        'error': result.error or "Unknown error"
                    })
                    if i < len(plan.steps) - 1:
                        break  # Re-plan on mid-execution failure

            if all(r.success for r in results):
                return self._build_response(results, session, prompt)

        return ExecuteResponse(
            status=TaskStatus.ERROR,
            success=False,
            error=f"Failed after {max_retries} attempts. Errors: {execution_errors}"
        )

    async def _execute_action(
        self,
        action: str,
        params: Dict[str, Any],
        thread_id: str,
        session: Session
    ) -> ExecuteResponse:
        """Execute direct action."""
        result = await self._execute_step_by_name(action, params, session, thread_id)
        return self._build_response([result], session, action)

    async def _execute_step_by_name(
        self,
        action: str,
        params: Dict[str, Any],
        session: Session,
        thread_id: str
    ) -> StepResult:
        """Execute step by action name."""
        if action == 'load_file':
            return await self._step_load_file(params, session, thread_id)
        elif action == 'export':
            df = self._get_current_df(session)
            return await self._step_export(params, df, session, thread_id) if df is not None else StepResult(action='export', success=False, error="No data loaded")
        elif action == 'process':
            df = self._get_current_df(session)
            return await self._step_process(params, df, session, thread_id) if df is not None else StepResult(action='process', success=False, error="No data loaded")
        else:
            # Route all other actions through process
            instruction = params.get('instruction') or params.get('question') or action
            df = self._get_current_df(session)
            return await self._step_process({'instruction': instruction}, df, session, thread_id) if df is not None else StepResult(action='process', success=False, error="No data loaded")

    async def _execute_step(
        self,
        step: StepPlan,
        session: Session,
        thread_id: str
    ) -> StepResult:
        """Execute a single step."""
        action = step.action
        params: Dict[str, Any] = step.params or {}

        if action == 'load_file':
            return await self._step_load_file(params, session, thread_id)
        elif action == 'export':
            df = self._get_current_df(session)
            return await self._step_export(params, df, session, thread_id) if df is not None else StepResult(action='export', success=False, error="No data loaded")
        elif action == 'process':
            df = self._get_current_df(session)
            return await self._step_process(params, df, session, thread_id) if df is not None else StepResult(action='process', success=False, error="No data loaded")
        else:
            # Route through process
            instruction = params.get('instruction') or params.get('question') or action
            df = self._get_current_df(session)
            return await self._step_process({'instruction': instruction}, df, session, thread_id) if df is not None else StepResult(action='process', success=False, error="No data loaded")

    def _get_current_df(self, session: Session) -> Optional[pd.DataFrame]:
        """Get current dataframe from session."""
        latest_id = session.get_latest_file_id()
        return session.dataframes.get(latest_id) if latest_id else None

    async def _step_load_file(
        self,
        params: Dict[str, Any],
        session: Session,
        thread_id: str
    ) -> StepResult:
        """Load a file."""
        try:
            file_path = params.get('file_path') or params.get('filename')
            content = params.get('content')

            df, detection = await self.client.load_file(
                file_path=file_path,
                content=content,
                filename=params.get('filename')
            )

            file_id = params.get('file_id', params.get('filename', 'file'))
            self.state.store_dataframe(thread_id, file_id, df, str(file_path or ''))

            return StepResult(
                action='load_file',
                success=True,
                result={'file_id': file_id, 'shape': df.shape, 'columns': df.columns.tolist()},
                df_modified=True
            )
        except Exception as e:
            return StepResult(action='load_file', success=False, error=str(e))

    async def _step_process(
        self,
        params: Dict[str, Any],
        df: Optional[pd.DataFrame],
        session: Session,
        thread_id: str
    ) -> StepResult:
        """LLM-powered data processing."""
        try:
            if df is None:
                return StepResult(action='process', success=False, error="No data loaded")

            instruction = params.get('instruction') or params.get('question') or ''
            df_context = await self.client.build_context(df, instruction)

            answer = await self.answer_question(instruction, df_context, session.get_recent_history())
            code = answer.get('code')

            if not code:
                return StepResult(
                    action='process',
                    success=True,
                    result={'answer': answer.get('answer', '')},
                    df_modified=False
                )

            # Execute generated code
            exec_globals: Dict[str, Any] = {
                '__builtins__': self.SAFE_BUILTINS,
                'df': df.copy() if df is not None else None,
                'pd': pd,
                'result': None
            }

            def save_spreadsheet(data: pd.DataFrame, filename: str) -> str:
                """Save dataframe to file."""
                if not filename.endswith(('.xlsx', '.csv')):
                    filename += '.xlsx'
                filepath = str(Path(self.client.storage_dir) / filename)
                if filename.endswith('.csv'):
                    data.to_csv(filepath, index=False)
                else:
                    data.to_excel(filepath, index=False)
                self.state.store_dataframe(thread_id, filename, data, filepath)
                return filepath

            exec_globals['save_spreadsheet'] = save_spreadsheet

            try:
                exec(code, exec_globals)
            except Exception as e:
                return StepResult(
                    action='process',
                    success=False,
                    error=f"Code execution failed: {e}. Code: {code}"
                )

            result_data = exec_globals.get('result') or exec_globals.get('df')
            df_modified = isinstance(result_data, pd.DataFrame) and not result_data.equals(df)

            if df_modified:
                file_id = session.get_latest_file_id()
                if file_id:
                    self.state.store_dataframe(thread_id, file_id, result_data)

            # Format result
            computed_answer = answer.get('answer', '')
            if isinstance(result_data, (int, float)):
                computed_answer = str(result_data)
            elif isinstance(result_data, pd.DataFrame):
                computed_answer = f"Result: {len(result_data)} rows"
                result_data = result_data.head(50).to_dict('records')

            return StepResult(
                action='process',
                success=True,
                result={
                    'answer': computed_answer,
                    'data': self._sanitize_for_json(result_data),
                    'code': code
                },
                df_modified=df_modified
            )

        except Exception as e:
            return StepResult(action='process', success=False, error=str(e))

    async def _step_export(
        self,
        params: Dict[str, Any],
        df: Optional[pd.DataFrame],
        session: Session,
        thread_id: str
    ) -> StepResult:
        """Export dataframe to file."""
        try:
            if df is None:
                return StepResult(action='export', success=False, error="No data loaded")

            filename = params.get('filename', 'export.xlsx')
            if not filename.endswith(('.xlsx', '.csv')):
                filename += '.xlsx'

            filepath = str(Path(self.client.storage_dir) / filename)

            if filename.endswith('.csv'):
                df.to_csv(filepath, index=False)
            else:
                df.to_excel(filepath, index=False)

            self.state.store_dataframe(thread_id, filename, df, filepath)

            return StepResult(
                action='export',
                success=True,
                result={'file_path': filepath, 'filename': filename, 'rows': len(df)},
                df_modified=False
            )
        except Exception as e:
            return StepResult(action='export', success=False, error=str(e))

    def _build_response(
        self,
        results: List[StepResult],
        session: Session,
        prompt: str  # pylint: disable=unused-argument
    ) -> ExecuteResponse:
        """Build final response from step results."""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        message = f"Completed {len(successful)} step(s)"
        if failed:
            message += f", {len(failed)} failed"

        # Build canvas display
        canvas_display = None
        file_id = session.get_latest_file_id()
        if file_id:
            df = session.dataframes.get(file_id)
            if df is not None:
                canvas_display = CanvasService.build_spreadsheet_view(
                    filename=file_id,
                    dataframe=df.head(100),
                    title=file_id
                )

        return ExecuteResponse(
            status=TaskStatus.COMPLETED,
            success=len(successful) > 0,
            message=message,
            result={
                'steps': len(results),
                'successful': len(successful),
                'failed': len(failed),
                'errors': [r.error for r in failed]
            },
            canvas_display=canvas_display
        )

    def _sanitize_for_json(self, obj: Any) -> Any:
        """Sanitize object for JSON serialization."""
        if isinstance(obj, pd.DataFrame):
            return obj.head(50).to_dict('records')
        elif isinstance(obj, pd.Series):
            return obj.head(50).to_dict()
        elif isinstance(obj, (list, tuple)):
            return [self._sanitize_for_json(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)

    async def _build_context(self, session: Session, prompt: str) -> Dict[str, Any]:
        """Build context for LLM."""
        file_id = session.get_latest_file_id()
        df = session.dataframes.get(file_id) if file_id else None

        context: Dict[str, Any] = {
            'has_data': df is not None,
            'columns': df.columns.tolist() if df is not None else [],
            'data_preview': df.head(5).to_dict('records') if df is not None else None,
            'history': session.get_recent_history()
        }

        if df is not None:
            context['data_preview'] = await self.client.build_context(df, prompt)

        return context
