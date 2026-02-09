"""
Spreadsheet Agent v3.0 - Main Agent

Central orchestrator for spreadsheet operations.
Unified /execute endpoint with LLM-powered task decomposition.
"""

import logging
import traceback
import os
import json
from typing import Dict, Any, Optional, List
import pandas as pd
import uuid

from .config import logger
from .schemas import (
    ExecuteRequest, ExecuteResponse, ExecutionPlan, StepResult,
    TaskStatus, FileInfo
)
from .state import session_state, Session
from .client import df_client, SmartDataResolver
from .llm import llm_client

# CMS Integration
import sys
from pathlib import Path
backend_root = Path(__file__).parent.parent.parent.resolve()
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from services.content_management_service import (
    ContentManagementService,
    ContentSource,
    ContentType,
    ContentPriority
)
from services.canvas_service import CanvasService
from backend.schemas import AgentResponse, StandardAgentResponse, AgentResponseStatus

logger = logging.getLogger("spreadsheet_agent.agent")


class SpreadsheetAgent:
    """
    Central orchestrator for spreadsheet operations.
    
    Features:
    - Unified /execute endpoint
    - LLM-powered task decomposition
    - Smart data resolution
    - Session management
    """
    
    def __init__(self):
        self.state = session_state
        self.client = df_client
        self.llm = llm_client
        self.resolver = SmartDataResolver(self.client, self.state)
        self.cms = ContentManagementService()
        
        logger.info("SpreadsheetAgent initialized")
        
    def _extract_prompt(self, params: Dict[str, Any]) -> Optional[str]:
        """
        Robustly extract prompt/instruction from parameters.
        Checks multiple common fields used by the Orchestrator.
        """
        if not params:
            return None
            
        # Priority order of fields to check
        fields = ['prompt', 'query', 'instruction', 'q', 'p', 'content', 'message']
        
        for field in fields:
            if params.get(field):
                return str(params[field])
                
        return None
    
    # ========================================================================
    # PUBLIC API
    # ========================================================================
    
    async def execute(
        self,
        prompt: str = None,
        action: str = None,
        params: Dict[str, Any] = None,
        thread_id: str = "default",
        task_id: str = None,
        file_content: bytes = None,
        filename: str = None
    ) -> AgentResponse:
        """
        Unified execution endpoint.
        
        Supports:
        1. Complex prompt mode: LLM decomposes into steps
        2. Direct action mode: Execute specific action
        3. File upload mode: Load file from content
        """
        params = params or {}
        
        try:
            # Get or create session
            session = self.state.get_or_create(thread_id)
            
            # ORCHESTRATOR COMPATIBILITY: Robust prompt extraction
            # The orchestrator may send instructions in various fields (query, instruction, etc.)
            if not prompt:
                prompt = self._extract_prompt(params)
                if prompt:
                    logger.info(f"Extracted prompt from params: {prompt[:50]}...")

            # Handle file upload if content provided
            # CRITICAL FIX: Load file first, but don't return immediately if prompt is also provided
            if file_content and filename:
                # Load the file into session
                upload_result = await self._handle_file_upload(
                    file_content, filename, thread_id, session
                )
                
                # If no prompt was provided, just return the upload result
                if not prompt:
                    return upload_result
                    
                # If upload failed, return the error
                if not upload_result.success:
                    return upload_result
                    
                # Otherwise, continue to process the prompt with the newly loaded file
                logger.info(f"File loaded successfully, now processing prompt: {prompt[:50]}...")
            
            # ORCHESTRATOR COMPATIBILITY: Auto-load from file_path in params
            # The orchestrator sends local file paths in params, we must load them!
            file_path = params.get('file_path')
            
            # DEBUG: Log what we received
            logger.info(f"[DEBUG] Params received: {list(params.keys())}")
            logger.info(f"[DEBUG] file_path from params: {file_path}")
            logger.info(f"[DEBUG] file_id from params: {params.get('file_id')}")
            
            # FALLBACK 1: Check if file_id is actually a full path
            if not file_path and params.get('file_id'):
                potential_path = params.get('file_id')
                if os.path.exists(potential_path):
                    file_path = potential_path
                    logger.info(f"[EXTRACT] file_id was actually a path: {file_path}")
            
            # FALLBACK 2: Extract file reference from prompt text if params is empty
            # The orchestrator sometimes embeds file info in prompt like: "file_id='...' or (file_id='...')"
            if not file_path and prompt:
                import re
                # Try to extract file_id/file_path from prompt text
                path_match = re.search(r"(?:file_path|path)=['\"]?([^'\")\s]+)['\"]?", prompt, re.IGNORECASE)
                if path_match:
                    potential_path = path_match.group(1)
                    if os.path.exists(potential_path):
                        file_path = potential_path
                        logger.info(f"[EXTRACT] Found file_path in prompt: {file_path}")
                
                if not file_path:
                    file_id_match = re.search(r"file_id=['\"]?([^'\")\s]+)['\"]?", prompt)
                    if file_id_match:
                        extracted_file_id = file_id_match.group(1)
                        logger.info(f"[EXTRACT] Found file_id in prompt: {extracted_file_id}")
                        
                        # Try to find this file in the storage directories
                        storage_dirs = [
                            "d:/Internship/Orbimesh/storage/spreadsheets",
                            "d:/Internship/Orbimesh/storage/spreadsheet_agent"
                        ]
                        for storage_dir in storage_dirs:
                            potential_path = os.path.join(storage_dir, extracted_file_id)
                            if os.path.exists(potential_path):
                                file_path = potential_path
                                logger.info(f"[EXTRACT] Found file at: {file_path}")
                                break
                            # Also try without extension matching
                            for ext in ['.xlsx', '.xls', '.csv']:
                                if not extracted_file_id.endswith(ext):
                                    potential_path = os.path.join(storage_dir, extracted_file_id + ext)
                                    if os.path.exists(potential_path):
                                        file_path = potential_path
                                        logger.info(f"[EXTRACT] Found file at: {file_path}")
                                        break
            
            # Now load the file if we have a path
            if file_path and not session.dataframes:
                try:
                    logger.info(f"Auto-loading local file from path: {file_path}")
                    df, detection = await self.client.load_file(file_path=file_path)
                    
                    # Use filename from path as ID
                    file_id = os.path.basename(file_path)
                    self.state.store_dataframe(thread_id, file_id, df, str(file_path))
                    logger.info(f"Successfully auto-loaded file: {file_id}, shape: {df.shape}")
                except Exception as e:
                    logger.error(f"Failed to auto-load file from path {file_path}: {e}")
                    return ExecuteResponse(
                        status=TaskStatus.ERROR,
                        success=False,
                        error=f"Failed to load required file: {e}"
                    )
            else:
                if not file_path:
                    logger.warning(f"[DEBUG] No file_path found in params or prompt")

            
            # Complex prompt mode
            if prompt and not action:
                 exec_res = await self._execute_complex(prompt, thread_id, session, params)
            
            # Direct action mode
            elif action:
                 exec_res = await self._execute_action(action, params, thread_id, session)
            
            # Auto-generated prompt if file exists but no instruction
            elif not prompt and session.get_latest_file_id():
                logger.info("No prompt/action provided but file exists in session. Defaulting to summary.")
                prompt = "Provide a comprehensive summary of this data including columns, rows, and key statistics."
                exec_res = await self._execute_complex(prompt, thread_id, session, params)
            
            # No valid input
            else:
                exec_res = ExecuteResponse(
                    status=TaskStatus.ERROR,
                    success=False,
                    error="Either 'prompt' or 'action' must be provided"
                )

            # --- STANDARDIZATION ADAPTER ---
            # Convert internal ExecuteResponse to StandardAgentResponse
            std_response = StandardAgentResponse(
                status="success" if exec_res.success else "error",
                summary=exec_res.message or ("Task completed" if exec_res.success else "Task failed"),
                data=exec_res.result or {},
                canvas_display=exec_res.canvas_display.model_dump() if exec_res.canvas_display else None,
                error_message=exec_res.error
            )
            
            return AgentResponse(
                status=AgentResponseStatus.COMPLETE if exec_res.success else AgentResponseStatus.ERROR,
                result=exec_res.dict(), # Keep full legacy result for now
                standard_response=std_response,
                error=exec_res.error
            )
            
        except Exception as e:
            logger.error(f"Execute failed: {e}\n{traceback.format_exc()}")
            return AgentResponse(
                status=AgentResponseStatus.ERROR,
                error=str(e),
                standard_response=StandardAgentResponse(
                    status="error",
                    summary="Execution failed hard",
                    error_message=str(e)
                )
            )
    
    async def continue_task(
        self,
        task_id: str,
        user_response: str,
        thread_id: str = "default"
    ) -> ExecuteResponse:
        """Resume a paused task with user input."""
        try:
            # Get paused task context
            paused = self.state.resume_task(thread_id, task_id)
            if not paused:
                return ExecuteResponse(
                    status=TaskStatus.ERROR,
                    success=False,
                    error=f"No paused task found with ID: {task_id}"
                )
            
            # Get original context
            context = paused.get('context', {})
            original_prompt = context.get('original_prompt', '')
            
            # Modify prompt with user response
            modified_prompt = f"{original_prompt}\n\nUser clarification: {user_response}"
            
            # Re-execute with clarification
            return await self.execute(
                prompt=modified_prompt,
                thread_id=thread_id,
                task_id=task_id
            )
            
        except Exception as e:
            logger.error(f"Continue failed: {e}")
            return ExecuteResponse(
                status=TaskStatus.ERROR,
                success=False,
                error=str(e)
            )
    
    # ========================================================================
    # COMPLEX PROMPT EXECUTION
    # ========================================================================
    
    async def _execute_complex(
        self,
        prompt: str,
        thread_id: str,
        session: Session,
        params: Dict[str, Any],
        max_step_retries: int = 3,
        max_plan_retries: int = 2
    ) -> ExecuteResponse:
        """
        LLM-powered task decomposition and execution with:
        - Per-step retries with error feedback
        - Dynamic plan re-evaluation on failures
        - Cumulative error learning
        """
        logger.info(f"[COMPLEX] Processing: {prompt[:100]}...")
        
        # Track errors across all attempts
        execution_errors = []
        plan_attempts = 0
        
        while plan_attempts < max_plan_retries:
            plan_attempts += 1
            
            
            # Build context with current state (with ACTUAL DATA!)
            context = await self._build_context(session, prompt=prompt)
            context['previous_errors'] = execution_errors if execution_errors else None
            
            # Decompose into steps (with error context if retrying)
            error_summary = None
            if execution_errors:
                error_summary = "Previous execution failed:\n" + "\n".join([
                    f"- Step '{e['step']}': {e['error']}" for e in execution_errors[-5:]
                ])
            
            plan = await self.llm.decompose_request(prompt, context, error_context=error_summary)
            
            # Check if clarification needed
            if plan.needs_clarification:
                task_id = f"task-{len(session.pending_tasks)}"
                self.state.pause_task(
                    thread_id, task_id,
                    plan.question,
                    {"original_prompt": prompt, "plan": plan.model_dump(), "errors": execution_errors}
                )
                
                return ExecuteResponse(
                    status=TaskStatus.NEEDS_INPUT,
                    success=True,
                    question=plan.question,
                    question_type="choice" if plan.options else "text",
                    options=plan.options,
                    context={"task_id": task_id}
                )
            
            # Execute steps with retries
            logger.info(f"[PLAN {plan_attempts}] Executing {len(plan.steps)} steps: {[s.action for s in plan.steps]}")
            
            results = []
            plan_failed = False
            
            # CRITICAL: Initialize current_df from session if data is already loaded
            # This ensures the auto-loaded dataframe is available for processing
            current_df = None
            latest_id = session.get_latest_file_id()
            if latest_id:
                current_df = session.dataframes.get(latest_id)
                if current_df is not None:
                    logger.info(f"[EXEC] Using already loaded dataframe: {latest_id}, shape: {current_df.shape}")
            
            for i, step in enumerate(plan.steps):
                step_result = await self._execute_step_with_retry(
                    step=step,
                    session=session,
                    current_df=current_df,
                    thread_id=thread_id,
                    max_retries=max_step_retries,
                    step_number=i + 1,
                    total_steps=len(plan.steps)
                )
                
                results.append(step_result)
                
                if step_result.success:
                    # Update current_df if step modified data
                    if step_result.df_modified:
                        latest_id = session.get_latest_file_id()
                        if latest_id:
                            current_df = session.dataframes.get(latest_id)
                    
                    session.add_operation(
                        step.action,
                        step.description or str(step.params),
                        {"success": True}
                    )
                else:
                    # Step failed after all retries
                    execution_errors.append({
                        'step': step.action,
                        'error': step_result.error,
                        'params': str(step.params)[:100]
                    })
                    
                    logger.warning(f"Step {step.action} failed after retries: {step_result.error}")
                    
                    # Check if we should re-evaluate the entire plan
                    if i < len(plan.steps) - 1:  # Not the last step
                        logger.info("Re-evaluating plan due to mid-execution failure...")
                        plan_failed = True
                        break
            
            if not plan_failed:
                # All steps completed (some may have failed)
                return self._build_response(results, session, prompt)
        
        # All plan attempts exhausted
        logger.error(f"All {max_plan_retries} plan attempts failed")
        return ExecuteResponse(
            status=TaskStatus.ERROR,
            success=False,
            error=f"Failed after {max_plan_retries} plan attempts. Errors: {execution_errors}",
            context={"errors": execution_errors}
        )
    
    async def _execute_step_with_retry(
        self,
        step,
        session: Session,
        current_df: pd.DataFrame,
        thread_id: str,
        max_retries: int,
        step_number: int,
        total_steps: int
    ) -> StepResult:
        """
        Execute a step with retry loop and error learning.
        """
        step_errors = []
        
        for attempt in range(1, max_retries + 1):
            logger.info(f"[STEP {step_number}/{total_steps}] {step.action} - Attempt {attempt}/{max_retries}")
            
            try:
                result = await self._execute_step(step, session, current_df, thread_id)
                
                if result.success:
                    if attempt > 1:
                        logger.info(f"Step {step.action} succeeded on attempt {attempt}")
                    return result
                else:
                    # Step returned failure (not exception)
                    step_errors.append({
                        'attempt': attempt,
                        'error': result.error
                    })
                    
                    if attempt < max_retries:
                        # Modify step params based on error
                        step = await self._adjust_step_for_retry(
                            step, result.error, step_errors, current_df
                        )
                        
            except Exception as e:
                step_errors.append({
                    'attempt': attempt,
                    'error': str(e),
                    'type': type(e).__name__
                })
                logger.warning(f"Step {step.action} attempt {attempt} exception: {e}")
                
                if attempt < max_retries:
                    # Try to adjust step for next attempt
                    step = await self._adjust_step_for_retry(
                        step, str(e), step_errors, current_df
                    )
        
        # All retries exhausted
        return StepResult(
            action=step.action,
            success=False,
            error=f"Failed after {max_retries} attempts: {step_errors[-1]['error']}",
            context={'all_errors': step_errors}
        )
    
    async def _adjust_step_for_retry(
        self,
        step,
        error: str,
        previous_errors: List[Dict],
        current_df: pd.DataFrame
    ):
        """
        Ask LLM to adjust step parameters based on error.
        """
        try:
            # Build context about the failure
            df_context = ""
            if current_df is not None:
                df_context = f"Current DataFrame columns: {current_df.columns.tolist()}"
            
            error_history = "\n".join([
                f"Attempt {e['attempt']}: {e['error']}" for e in previous_errors
            ])
            
            # Ask LLM for adjusted step
            adjusted = await self.llm.adjust_step(
                action=step.action,
                original_params=step.params,
                error=error,
                error_history=error_history,
                df_context=df_context
            )
            
            if adjusted:
                from .schemas import StepPlan
                return StepPlan(
                    action=adjusted.get('action', step.action),
                    params=adjusted.get('params', step.params),
                    description=f"Adjusted: {adjusted.get('reasoning', 'retry')}"
                )
        except Exception as e:
            logger.warning(f"Could not adjust step: {e}")
        
        return step  # Return original if adjustment fails
    
    async def _execute_step(
        self,
        step,
        session: Session,
        current_df: pd.DataFrame,
        thread_id: str
    ) -> StepResult:
        """Execute a single step from the plan."""
        action = step.action
        params = step.params
        
        # Smart data resolution if needed
        if current_df is None and action not in ['load_file', 'create']:
            try:
                current_df = await self.resolver.resolve_dataframe(params, thread_id, require_data=False)
            except:
                pass
        
        # Route to appropriate handler - SIMPLIFIED ARCHITECTURE
        # Only 3 core actions: load_file, process, export
        # Everything else routes through process (LLM-powered)
        
        if action == 'load_file':
            return await self._step_load_file(params, session, thread_id)
        elif action == 'export':
            return await self._step_export(params, current_df, session, thread_id)
        elif action == 'process':
            # The core LLM-powered action - full pandas freedom
            return await self._step_process(params, current_df, session, thread_id)
        else:
            # ALL other actions route through process for maximum flexibility
            # This includes: query, filter, sort, aggregate, add_column, drop_column,
            # rename_column, fill_na, transform, compare, merge, etc.
            instruction = params.get('instruction') or params.get('question') or str(params)
            if action not in ['process']:
                instruction = f"{action}: {instruction}"
            return await self._step_process(
                {'instruction': instruction},
                current_df, session, thread_id
            )
    
    # ========================================================================
    # STEP HANDLERS
    # ========================================================================
    
    async def _step_load_file(
        self,
        params: Dict,
        session: Session,
        thread_id: str
    ) -> StepResult:
        """Load a file."""
        try:
            file_path = params.get('file_path') or params.get('filename')
            content = params.get('content')
            
            # Fallback: If no file_path but we have a filename/id, try storage dir
            if not file_path and not content:
                candidate = params.get('filename') or params.get('file_id')
                
                # CMS Integration: Try fetching from CMS first
                if candidate and self.cms:
                    try:
                        # Try to get content from CMS
                        logger.info(f"Attempting to fetch {candidate} from CMS...")
                        # get_content returns (metadata, content)
                        meta, cms_content = self.cms.get_content(candidate) # candidate as ID
                        if cms_content:
                            logger.info(f"✅ Fetched content {candidate} from CMS (Size: {len(cms_content)} bytes)")
                            content = cms_content
                            # Update filename if available from metadata
                            if meta and meta.name:
                                params['filename'] = meta.name
                    except Exception as cms_err:
                        logger.warning(f"CMS fetch failed for {candidate}: {cms_err}") # Not fatal, continue to local check

                if not content and candidate:
                    # Clean filename (remove potential path components)
                    candidate_name = os.path.basename(candidate)
                    potential_path = self.client.storage_dir / candidate_name
                    if potential_path.exists():
                        file_path = str(potential_path)
                        logger.info(f"Using fallback file path from storage: {file_path}")
                    elif candidate != candidate_name:
                         # Try full path if provided
                         if os.path.exists(candidate):
                             file_path = candidate
                             logger.info(f"Using provided candidate as path: {file_path}")

            # Safety check
            if not file_path and not content:
                 # Last resort: check if file_id matches a known file in state?
                 # Handled by state retrieval usually, but load_file implies new load.
                 pass
            
            df, detection_info = await self.client.load_file(
                file_path=file_path,
                content=content,
                filename=params.get('filename')
            )
            
            # Store in session
            file_id = params.get('file_id', params.get('filename', 'file'))
            self.state.store_dataframe(thread_id, file_id, df, str(file_path or ''))
            
            return StepResult(
                action='load_file',
                success=True,
                result={
                    'file_id': file_id,
                    'shape': df.shape,
                    'columns': df.columns.tolist(),
                    'detection': detection_info
                },
                df_modified=True
            )
        except Exception as e:
            return StepResult(action='load_file', success=False, error=str(e))
    
    async def _step_process(
        self,
        params: Dict,
        df: pd.DataFrame,
        session: Session,
        thread_id: str
    ) -> StepResult:
        """
        Unified LLM-powered data processing - handles ALL data operations.
        
        This is the core method that gives the LLM complete pandas freedom.
        It can handle: queries, aggregations, filters, sorts, transforms,
        column operations, calculations, and any other pandas operation.
        """
        try:
            instruction = params.get('instruction') or params.get('question') or ''
            
            if df is None:
                return StepResult(action='process', success=False, error="No data loaded")
            
            # Build rich context for the LLM
            df_context = await self.client.build_context(df, instruction)
            
            # Generate pandas code with full freedom
            logger.info(f"[_step_process] Instruction: {instruction}")
            answer = await self.llm.answer_question(
                instruction,
                df_context,
                session.get_recent_history()
            )
            
            # Execute the generated code
            result_data = None
            computed_answer = None
            df_modified = False
            code = answer.get('code')
            
            if code:
                logger.info(f"[_step_process] Executing code: {code}")
            # Track created files to return the ID for canvas
            created_files = []
            
            # Define safe file saving helper
            def save_spreadsheet(data, filename):
                import os
                
                # backend/agents/spreadsheet_agent/agent.py -> backend/agents -> backend -> Orbimesh
                # We want Orbimesh/storage/spreadsheet_agent
                
                # Use relative path logic precisely
                root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
                storage_dir = os.path.join(root_dir, 'storage', 'spreadsheet_agent')
                os.makedirs(storage_dir, exist_ok=True)
                
                if not any(filename.lower().endswith(ext) for ext in ['.xlsx', '.csv']):
                     filename += '.xlsx'
                
                filepath = os.path.join(storage_dir, filename)
                
                if hasattr(data, 'head'): # DataFrame or Series
                    if filename.lower().endswith('.csv'):
                        data.to_csv(filepath, index=False)
                    else:
                        data.to_excel(filepath, index=False)
                    
                    # CRITICAL FIX: Register the new file in session state!
                    # This makes it the "latest" file so it gets displayed on canvas.
                    if 'thread_id' in locals() or 'thread_id' in globals():
                        # thread_id is captured from outer scope (_step_process)
                        self.state.store_dataframe(thread_id, filename, data, filepath)
                        logger.info(f"Registered new file in session: {filename}")
                        created_files.append({
                            "file_name": filename,
                            "file_path": filepath,
                            "file_type": "spreadsheet",
                            "file_id": filename
                        }) # Track it properly!
                else:
                     raise ValueError("Data must be a pandas DataFrame or Series")
                
                # Make relative to project root for display
                try:
                    return filepath 
                except:
                    return filepath

            # Use combined scope to avoid closure/scope issues
            exec_globals = {
                '__builtins__': self.SAFE_BUILTINS,
                'df': df.copy() if df is not None else None,
                'pd': pd,
                'save_spreadsheet': save_spreadsheet,
                'result': None # Explicitly init result
            }
                
            try:
                if code:
                    exec(code, exec_globals)
                
                # Check for explicit result or modified df
                result_data = exec_globals.get('result')
                
                # Intelligent DataFrame detection:
                # If 'result' wasn't set, look for ANY new pandas Local variable that is a DataFrame
                if result_data is None:
                     # Check if 'df' was modified in place?
                     if exec_globals.get('df') is not None and not exec_globals.get('df').equals(df):
                          result_data = exec_globals.get('df')
                     else:
                          # Look for other DataFrames created in the scope (e.g. new_df)
                          # We prefer variables that look like "df" or "result" or "output"
                          candidates = []
                          for k, v in exec_globals.items():
                               if k not in ['df', 'pd', 'save_spreadsheet', 'result', '__builtins__'] and not k.startswith('_'):
                                    if isinstance(v, pd.DataFrame):
                                         candidates.append(k)
                          
                          if candidates:
                               # Pick the last defined candidate (heuristic) or prefers specific names
                               logger.info(f"[_step_process] Found new DataFrames in scope: {candidates}")
                               # Use the last one found as the result
                               result_data = exec_globals[candidates[-1]]
                               # Update the main df reference for subsequent logic
                               exec_globals['df'] = result_data

                if result_data is None:
                     result_data = exec_globals.get('df')

                
                # Check if the dataframe was modified
                # CRITICAL FIX: Treat 'result_data' as the potentially modified dataframe if it is a DataFrame
                # This handles cases where code does `result = df[...]` instead of modifying `df` in-place
                modified_df = result_data if isinstance(result_data, pd.DataFrame) else exec_globals.get('df')
                
                if modified_df is not None and isinstance(modified_df, pd.DataFrame):
                    # Check if it differs from input df
                    if not modified_df.equals(df):
                        df_modified = True
                        # Store the modified dataframe
                        # overwrite the latest file in session essentially updating the state
                        file_id = session.get_latest_file_id()
                        if file_id:
                            self.state.store_dataframe(thread_id, file_id, modified_df)
                            logger.info(f"[_step_process] State updated: {file_id} modified (cols: {len(modified_df.columns)})")
                
                # Handle different result types
                if isinstance(result_data, (int, float)):
                    computed_answer = str(result_data)
                    logger.info(f"[_step_process] Scalar result: {computed_answer}")
                elif isinstance(result_data, str) and result_data.strip():
                    computed_answer = result_data
                    logger.info(f"[_step_process] String result: {computed_answer}")
                elif hasattr(result_data, 'item'):  # numpy scalar
                    computed_answer = str(result_data.item())
                    logger.info(f"[_step_process] Numpy scalar: {computed_answer}")
                elif isinstance(result_data, pd.DataFrame):
                    if len(result_data) <= 50:
                        result_data = result_data.head(50).to_dict('records') # sanitize later
                    else:
                        result_data = result_data.head(50).to_dict('records')
                    if result_data:
                        computed_answer = self._format_dataframe_answer(result_data, instruction)
                elif isinstance(result_data, pd.Series):
                    if len(result_data) <= 20:
                        result_data = result_data.to_dict()
                        computed_answer = str(result_data)
                    else:
                        computed_answer = f"Series with {len(result_data)} items"
                        result_data = result_data.head(20).to_dict()
                elif isinstance(result_data, (dict, list, tuple)):
                    try:
                        # Format pretty string
                        formatted = json.dumps(self._sanitize_for_json(result_data), indent=2)
                        if len(formatted) > 3000:
                            formatted = formatted[:3000] + "\\n... (truncated)"
                        computed_answer = f"Result:\\n{formatted}"
                    except:
                            computed_answer = str(result_data)
                    logger.info(f"[_step_process] Structured result len: {len(computed_answer)}")
                
                logger.info(f"[_step_process] Execution success, modified={df_modified}")
                
            except Exception as e:
                logger.error(f"[_step_process] Code execution failed: {e}")
                logger.error(f"[_step_process] Code was: {code}")
                # Return the error so LLM can retry with different approach
                return StepResult(
                    action='process',
                    success=False,
                    error=f"Code execution failed: {e}. Code: {code}"
                )
            
            # Build the final answer
            final_answer = computed_answer if computed_answer else answer.get('answer', '')
            logger.info(f"[_step_process] Final answer: {final_answer[:200] if final_answer else 'None'}...")
            
            result_payload = {
                'answer': final_answer,
                'data': self._sanitize_for_json(result_data),
                'code': answer.get('code'),
                'confidence': answer.get('confidence', 0.8)
            }
            
            if created_files:
                result_payload['file_id'] = created_files[-1]
                result_payload['generated_files'] = created_files # Inform orchestrator state management
                logger.info(f"[_step_process] Returning result with file_id: {created_files[-1]} and {len(created_files)} generated files")
            
            # CMS Registration: Register generated files
            if created_files and self.cms:
                for cf in created_files:
                    try:
                        # cf has file_path, file_name, file_id
                        f_path = cf.get('file_path')
                        f_name = cf.get('file_name')
                        if f_path and os.path.exists(f_path):
                            # Read content to register
                            with open(f_path, 'rb') as f:
                                f_bytes = f.read()
                            
                            # Register with CMS (async but inside sync loop - problematic? 
                            # No, _step_process is async, so we can await!)
                            await self.cms.register_content(
                                content=f_bytes,
                                name=f_name,
                                source=ContentSource.AGENT_OUTPUT,
                                content_type=ContentType.SPREADSHEET,
                                priority=ContentPriority.MEDIUM,
                                tags=["spreadsheet_agent", "generated"],
                                thread_id=thread_id
                            )
                            logger.info(f"✅ Registered generated file {f_name} with CMS")
                    except Exception as cms_reg_err:
                         logger.error(f"Failed to register {cf.get('file_name')} with CMS: {cms_reg_err}")
            
            return StepResult(
                action='process',
                success=True,
                result=result_payload,
                df_modified=df_modified
            )
            
        except Exception as e:
            logger.error(f"[_step_process] Error: {e}")
            return StepResult(action='process', success=False, error=str(e))
    
    
    def _format_dataframe_answer(self, data: list, question: str) -> str:
        """Format DataFrame results into a readable answer string."""
        if not data:
            return "No results found."
        
        # For small result sets, include the data in the answer
        if len(data) <= 10:
            # Get column names
            columns = list(data[0].keys()) if data else []
            
            # Build a simple table representation
            lines = []
            for i, row in enumerate(data, 1):
                row_parts = [f"{k}: {v}" for k, v in row.items()]
                lines.append(f"{i}. " + ", ".join(row_parts))
            
            return "\n".join(lines)
        else:
            # For larger results, just summarize
            return f"Found {len(data)} results. Showing first {min(len(data), 50)} in the data field."
    
    
    async def _step_filter(
        self,
        params: Dict,
        df: pd.DataFrame,
        session: Session,
        thread_id: str
    ) -> StepResult:
        """Filter rows."""
        try:
            column = params.get('column')
            operator = params.get('operator', '==')
            value = params.get('value')
            
            if df is None:
                return StepResult(action='filter', success=False, error="No data loaded")
            
            # Resolve column name
            cols = self.resolver.resolve_columns(df, [column])
            if not cols:
                return StepResult(action='filter', success=False, error=f"Column not found: {column}")
            
            col = cols[0]
            
            # Apply filter
            if operator == '==':
                filtered = df[df[col] == value]
            elif operator == '!=':
                filtered = df[df[col] != value]
            elif operator == '>':
                filtered = df[df[col] > value]
            elif operator == '<':
                filtered = df[df[col] < value]
            elif operator == '>=':
                filtered = df[df[col] >= value]
            elif operator == '<=':
                filtered = df[df[col] <= value]
            elif operator == 'contains':
                filtered = df[df[col].astype(str).str.contains(str(value), case=False, na=False)]
            else:
                filtered = df[df[col] == value]
            
            # Store result
            file_id = session.get_latest_file_id()
            if file_id:
                self.state.store_dataframe(thread_id, file_id, filtered)
            
            return StepResult(
                action='filter',
                success=True,
                result={
                    'original_rows': len(df),
                    'filtered_rows': len(filtered),
                    'removed': len(df) - len(filtered),
                    'data': filtered.head(100).to_dict('records')  # Include actual filtered data
                },
                df_modified=True
            )
        except Exception as e:
            return StepResult(action='filter', success=False, error=str(e))
    
    async def _step_sort(
        self,
        params: Dict,
        df: pd.DataFrame,
        session: Session,
        thread_id: str
    ) -> StepResult:
        """Sort data."""
        try:
            column = params.get('column')
            ascending = params.get('ascending', True)
            
            if df is None:
                return StepResult(action='sort', success=False, error="No data loaded")
            
            cols = self.resolver.resolve_columns(df, [column])
            if not cols:
                return StepResult(action='sort', success=False, error=f"Column not found: {column}")
            
            sorted_df = df.sort_values(by=cols[0], ascending=ascending)
            
            file_id = session.get_latest_file_id()
            if file_id:
                self.state.store_dataframe(thread_id, file_id, sorted_df)
            
            return StepResult(
                action='sort',
                success=True,
                result={'sorted_by': cols[0], 'ascending': ascending},
                df_modified=True
            )
        except Exception as e:
            return StepResult(action='sort', success=False, error=str(e))
    
    async def _step_aggregate(
        self,
        params: Dict,
        df: pd.DataFrame,
        session: Session,
        thread_id: str
    ) -> StepResult:
        """Group and aggregate data."""
        try:
            group_by = params.get('group_by')
            column = params.get('column')
            function = params.get('function', 'sum')
            
            if df is None:
                return StepResult(action='aggregate', success=False, error="No data loaded")
            
            # Resolve columns
            group_cols = []
            if group_by:
                group_cols = self.resolver.resolve_columns(df, [group_by] if isinstance(group_by, str) else group_by)
                if not group_cols:
                     return StepResult(action='aggregate', success=False, error=f"Group column not found: {group_by}")

            agg_cols = self.resolver.resolve_columns(df, [column])
            if not agg_cols and column:
                 return StepResult(action='aggregate', success=False, error=f"Aggregate column not found: {column}")
            
            # Perform aggregation
            if agg_cols:
                target_col = agg_cols[0]
                if group_cols:
                    # Grouped aggregation
                    if function == 'sum':
                        result = df.groupby(group_cols)[target_col].sum().reset_index()
                    elif function == 'mean':
                        result = df.groupby(group_cols)[target_col].mean().reset_index()
                    elif function == 'count':
                        result = df.groupby(group_cols)[target_col].count().reset_index()
                    elif function in ('nunique', 'count_unique', 'unique_count'):
                        result = df.groupby(group_cols)[target_col].nunique().reset_index()
                    elif function == 'min':
                        result = df.groupby(group_cols)[target_col].min().reset_index()
                    elif function == 'max':
                        result = df.groupby(group_cols)[target_col].max().reset_index()
                    else:
                        result = df.groupby(group_cols)[target_col].sum().reset_index()
                else:
                    # Global aggregation (no group by)
                    if function == 'sum':
                        val = df[target_col].sum()
                    elif function == 'mean':
                        val = df[target_col].mean()
                    elif function == 'count':
                        val = df[target_col].count()
                    elif function in ('nunique', 'count_unique', 'unique_count'):
                        val = df[target_col].nunique()
                    elif function == 'min':
                        val = df[target_col].min()
                    elif function == 'max':
                        val = df[target_col].max()
                    else:
                        val = df[target_col].sum()
                    
                    # Create a simple result dataframe
                    result = pd.DataFrame([{function: val}])
                    
                    # Optimization: Return scalar answer directly if it's a global aggregation
                    return StepResult(
                        action='aggregate',
                        success=True,
                        result={
                            'answer': f"The {function} of {target_col} is {val}",
                            'value': float(val) if hasattr(val, 'item') else val,
                            'data': result.to_dict('records')
                        },
                        df_modified=False # Don't update main df context for simple scalar queries
                    )

            else:
                # Count only if no agg column
                if group_cols:
                    result = df.groupby(group_cols).size().reset_index(name='count')
                else:
                    val = len(df)
                    return StepResult(
                        action='aggregate',
                        success=True,
                        result={
                             'answer': f"Count is {val}",
                             'value': val
                        }
                    )
            
            # For grouped results, we generally want to store them
            file_id = f"agg_{uuid.uuid4().hex[:8]}"
            self.state.store_dataframe(thread_id, file_id, result)

            return StepResult(
                action='aggregate',
                success=True,
                result={
                    'file_id': file_id,
                    'groups': len(result),
                    'data': result.head(100).to_dict('records') # Limit return size
                },
                df_modified=True
            )
        except Exception as e:
            return StepResult(action='aggregate', success=False, error=str(e))
    
    async def _step_add_column(
        self,
        params: Dict,
        df: pd.DataFrame,
        session: Session,
        thread_id: str
    ) -> StepResult:
        """Add a calculated column."""
        try:
            name = params.get('name')
            expression = params.get('expression')
            
            if df is None:
                return StepResult(action='add_column', success=False, error="No data loaded")
            
            # Generate code if expression is natural language
            if not expression.startswith('df['):
                code = await self.llm.generate_pandas_code(
                    f"Add a new column called '{name}' where: {expression}",
                    await self.client.build_context(df)
                )
            else:
                code = f"df['{name}'] = {expression}"
            
            # Execute
            exec_globals = {
                '__builtins__': self.SAFE_BUILTINS,
                'df': df.copy(),
                'pd': pd
            }
            exec(code, exec_globals)
            result_df = exec_globals['df']
            
            file_id = session.get_latest_file_id()
            if file_id:
                self.state.store_dataframe(thread_id, file_id, result_df)
            
            return StepResult(
                action='add_column',
                success=True,
                result={'column': name, 'code': code},
                df_modified=True
            )
        except Exception as e:
            return StepResult(action='add_column', success=False, error=str(e))
    
    async def _step_drop_column(
        self,
        params: Dict,
        df: pd.DataFrame,
        session: Session,
        thread_id: str
    ) -> StepResult:
        """Drop a column."""
        try:
            column = params.get('column')
            
            if df is None:
                return StepResult(action='drop_column', success=False, error="No data loaded")
            
            cols = self.resolver.resolve_columns(df, [column])
            if not cols:
                return StepResult(action='drop_column', success=False, error=f"Column not found: {column}")
            
            result_df = df.drop(columns=cols)
            
            file_id = session.get_latest_file_id()
            if file_id:
                self.state.store_dataframe(thread_id, file_id, result_df)
            
            return StepResult(
                action='drop_column',
                success=True,
                result={'dropped': cols[0]},
                df_modified=True
            )
        except Exception as e:
            return StepResult(action='drop_column', success=False, error=str(e))
    
    # Safe builtins for code execution
    SAFE_BUILTINS = {
        'print': print, 'len': len, 'sum': sum, 'min': min, 'max': max,
        'abs': abs, 'round': round, 'sorted': sorted, 'list': list,
        'dict': dict, 'str': str, 'int': int, 'float': float, 'bool': bool,
        'range': range, 'enumerate': enumerate, 'zip': zip,
        'True': True, 'False': False, 'None': None,
        '__import__': __import__,
        # Additional commonly needed builtins for pandas operations
        'isinstance': isinstance, 'type': type, 'tuple': tuple, 'set': set,
        'any': any, 'all': all, 'map': map, 'filter': filter,
        'hasattr': hasattr, 'getattr': getattr, 'setattr': setattr,
        'slice': slice, 'reversed': reversed, 'iter': iter, 'next': next,
        'callable': callable, 'repr': repr, 'format': format,
    }

    async def _step_transform(
        self,
        params: Dict,
        df: pd.DataFrame,
        session: Session,
        thread_id: str
    ) -> StepResult:
        """Apply a custom transformation via LLM."""
        try:
            instruction = params.get('instruction')
            
            if df is None:
                return StepResult(action='transform', success=False, error="No data loaded")
            
            # Generate code
            code = await self.llm.generate_pandas_code(
                instruction,
                await self.client.build_context(df)
            )
            
            # Execute
            # Execute
            exec_globals = {
                '__builtins__': self.SAFE_BUILTINS,
                'df': df.copy(),
                'pd': pd
            }
            exec(code, exec_globals)
            result_df = exec_globals['df']
            
            file_id = session.get_latest_file_id()
            if file_id:
                self.state.store_dataframe(thread_id, file_id, result_df)
            
            return StepResult(
                action='transform',
                success=True,
                result={
                    'code': code,
                    'shape': result_df.shape
                },
                df_modified=True
            )
        except Exception as e:
            return StepResult(action='transform', success=False, error=str(e))
    
    async def _step_export(
        self,
        params: Dict,
        df: pd.DataFrame,
        session: Session,
        thread_id: str
    ) -> StepResult:
        """Export data to file."""
        try:
            filename = params.get('filename', 'export.csv')
            format = params.get('format', 'csv')
            
            if df is None:
                return StepResult(action='export', success=False, error="No data loaded")
            
            file_id, file_path = await self.client.save_file(
                df, filename, format, thread_id
            )
            
            return StepResult(
                action='export',
                success=True,
                result={
                    'file_id': file_id,
                    'file_path': file_path,
                    'format': format,
                    'rows': len(df)
                }
            )
        except Exception as e:
            return StepResult(action='export', success=False, error=str(e))
    
    async def _step_merge(
        self,
        params: Dict,
        df: pd.DataFrame,
        session: Session,
        thread_id: str
    ) -> StepResult:
        """Merge/Join files."""
        try:
            # params: file_ids (list), how, on (optional)
            file_ids = params.get('file_ids', [])
            how = params.get('how', 'inner')
            on = params.get('on')
            
            # If current df is one of them, use it
            dfs_to_merge = []
            
            # Resolve dataframes
            for fid in file_ids:
                if fid in session.dataframes:
                    dfs_to_merge.append(session.dataframes[fid])
            
            # If current df is not in list but we have it, maybe add it?
            # Usually file_ids should specify exactly what to merge
            
            if len(dfs_to_merge) < 2:
                # Try to use current_df if available
                if df is not None and len(dfs_to_merge) == 1:
                     # Check if current_df is already in dfs_to_merge (by identity or content)
                     # Simpler: just append and assume user meant to merge current with another
                     dfs_to_merge.insert(0, df)
                
            if len(dfs_to_merge) < 2:
                return StepResult(action='merge', success=False, error="Need at least 2 DataFrames to merge")
            
            # Perform merge
            left = dfs_to_merge[0]
            right = dfs_to_merge[1]
            
            if on:
                result_df = pd.merge(left, right, how=how, on=on)
            else:
                # Let pandas infer or merge on index if no common columns?
                # Safer to let pandas infer
                result_df = pd.merge(left, right, how=how)
                
            # Store result
            file_id = f"merged_{uuid.uuid4().hex[:8]}"
            self.state.store_dataframe(thread_id, file_id, result_df)
            
            return StepResult(
                action='merge',
                success=True,
                result={
                    'file_id': file_id,
                    'rows': len(result_df),
                    'columns': len(result_df.columns)
                },
                df_modified=True
            )
        except Exception as e:
            return StepResult(action='merge', success=False, error=str(e))

    async def _step_rename_column(
        self,
        params: Dict,
        df: pd.DataFrame,
        session: Session,
        thread_id: str
    ) -> StepResult:
        """Rename a column."""
        try:
            old_name = params.get('old_name')
            new_name = params.get('new_name')
            
            if df is None:
                return StepResult(action='rename_column', success=False, error="No data loaded")
            
            cols = self.resolver.resolve_columns(df, [old_name])
            if not cols:
                return StepResult(action='rename_column', success=False, error=f"Column not found: {old_name}")
            
            result_df = df.rename(columns={cols[0]: new_name})
            
            file_id = session.get_latest_file_id()
            if file_id:
                self.state.store_dataframe(thread_id, file_id, result_df)
            
            return StepResult(
                action='rename_column',
                success=True,
                result={'old': cols[0], 'new': new_name},
                df_modified=True
            )
        except Exception as e:
            return StepResult(action='rename_column', success=False, error=str(e))

    async def _step_fill_na(
        self,
        params: Dict,
        df: pd.DataFrame,
        session: Session,
        thread_id: str
    ) -> StepResult:
        """Fill missing values."""
        try:
            column = params.get('column')
            value = params.get('value', 0)
            
            if df is None:
                return StepResult(action='fill_na', success=False, error="No data loaded")
            
            if column:
                cols = self.resolver.resolve_columns(df, [column])
                if not cols:
                    return StepResult(action='fill_na', success=False, error=f"Column not found: {column}")
                
                df_copy = df.copy()
                df_copy[cols[0]] = df_copy[cols[0]].fillna(value)
                result_df = df_copy
            else:
                # Fill all
                result_df = df.fillna(value)
            
            file_id = session.get_latest_file_id()
            if file_id:
                self.state.store_dataframe(thread_id, file_id, result_df)
            
            return StepResult(
                action='fill_na',
                success=True,
                result={'filled': True, 'value': value},
                df_modified=True
            )
        except Exception as e:
            return StepResult(action='fill_na', success=False, error=str(e))

    # ========================================================================
    # DIRECT ACTION EXECUTION
    # ========================================================================
    
    async def _execute_action(
        self,
        action: str,
        params: Dict[str, Any],
        thread_id: str,
        session: Session
    ) -> ExecuteResponse:
        """Execute a specific action directly."""
        try:
            # Resolve dataframe
            df = await self.resolver.resolve_dataframe(params, thread_id, require_data=False)
            
            # Create step and execute
            from .schemas import StepPlan
            step = StepPlan(action=action, params=params)
            result = await self._execute_step(step, session, df, thread_id)
            
            session.add_operation(action, str(params), result.result)
            
            return ExecuteResponse(
                status=TaskStatus.COMPLETE if result.success else TaskStatus.ERROR,
                success=result.success,
                result=result.result,
                error=result.error
            )
            
        except Exception as e:
            return ExecuteResponse(
                status=TaskStatus.ERROR,
                success=False,
                error=str(e)
            )
    
    # ========================================================================
    # FILE UPLOAD
    # ========================================================================
    
    async def _handle_file_upload(
        self,
        content: bytes,
        filename: str,
        thread_id: str,
        session: Session
    ) -> ExecuteResponse:
        """Handle file upload."""
        try:
            # Load file
            df, detection_info = await self.client.load_file(
                content=content,
                filename=filename
            )
            
            # Save to storage
            file_id, file_path = await self.client.save_file(
                df, filename, thread_id=thread_id
            )
            
            # Store in session
            self.state.store_dataframe(thread_id, file_id, df, file_path)
            
            session.add_operation('upload', filename, {'shape': df.shape})
            
            # Build canvas display
            canvas = self._build_canvas(df, f"Uploaded: {filename}", file_id)
            
            return ExecuteResponse(
                status=TaskStatus.COMPLETE,
                success=True,
                result={
                    'file_id': file_id,
                    'filename': filename,
                    'rows': len(df),
                    'columns': len(df.columns),
                    'column_names': df.columns.tolist(),
                    'detection': detection_info
                },
                canvas_display=canvas
            )
            
        except Exception as e:
            return ExecuteResponse(
                status=TaskStatus.ERROR,
                success=False,
                error=f"Upload failed: {e}"
            )
    
    # ========================================================================
    # HELPERS
    # ========================================================================
    
    async def _build_context(self, session: Session, prompt: str = None) -> Dict[str, Any]:
        """
        Build rich context for LLM from session with ACTUAL DATA PREVIEW.
        
        This is the critical context used for task decomposition.
        The LLM needs to SEE the data to make intelligent decisions!
        """
        has_data = bool(session.dataframes)
        columns = []
        data_preview = None
        
        if has_data:
            latest_id = session.get_latest_file_id()
            if latest_id:
                # Get the actual DataFrame
                df = session.dataframes.get(latest_id)
                
                if df is not None:
                    # Get column names
                    columns = df.columns.tolist()
                    
                    # Get ACTUAL DATA PREVIEW using our smart context builder!
                    # This shows the LLM real data, not just column names
                    try:
                        data_preview = await self.client.build_context(df, query=prompt)
                        logger.info(f"Built rich data context: {len(df)} rows, {len(columns)} columns")
                        logger.info(f"[CONTEXT] Data preview length: {len(data_preview) if data_preview else 0} chars")
                    except Exception as e:
                        logger.warning(f"Failed to build data preview: {e}")
                        # Fallback to basic info
                        data_preview = f"DataFrame with {len(df)} rows and {len(columns)} columns\nColumns: {columns}"
        
        return {
            'has_data': has_data,
            'columns': columns,
            'data_preview': data_preview,  # NEW: Actual data for LLM to see!
            'history': session.get_recent_history()
        }
    
    def _build_response(
        self,
        results: List[StepResult],
        session: Session,
        original_prompt: str
    ) -> ExecuteResponse:
        """Build final response from step results."""
        all_success = all(r.success for r in results)
        
        # Combine results with data truncation for lightweight transport
        safe_results = []
        for r in results:
            res_copy = r.result.copy() if isinstance(r.result, dict) else r.result
            if isinstance(res_copy, dict) and 'data' in res_copy:
                 # Truncate data in the log to prevent flooding
                 data = res_copy['data']
                 if isinstance(data, list) and len(data) > 3:
                     res_copy['data'] = data[:3] + [f"... {len(data)-3} more items ..."]
            
            safe_results.append({
                'action': r.action,
                'success': r.success,
                'result': res_copy,
                'error': r.error
            })

        combined_result = {
            'prompt': original_prompt,
            'steps_executed': len(results),
            'results': safe_results
        }
        
        # Get last successful result for primary answer
        canvas_df = None
        canvas_title = "Result"
        canvas_file_id = None
        
        # Check if ANY step modified the dataframe
        any_df_modified = any(r.df_modified for r in results if r.success)
        
        summary_parts = []
        
        # If any step modified data, the session's latest file IS the source of truth
        if any_df_modified:
            latest_id = session.get_latest_file_id()
            if latest_id:
                canvas_df = session.dataframes.get(latest_id)
                canvas_file_id = latest_id
                canvas_title = "Result Data"
                logger.info(f"[_build_response] Using latest modified session file: {latest_id}")

        for r in reversed(results):
            if r.success and r.result:
                if 'answer' in r.result:
                    combined_result['answer'] = r.result['answer']
                    summary_parts.append(f"Answer: {r.result['answer']}")
                
                # Check for explicit file_id in result (if we haven't already found one via modification)
                if canvas_df is None and 'file_id' in r.result:
                    canvas_file_id = r.result['file_id']
                    if canvas_file_id in session.dataframes:
                        canvas_df = session.dataframes[canvas_file_id]
                        if r.action == 'aggregate':
                            canvas_title = "Aggregation Result"
                            summary_parts.append(f"Aggregated data into {len(canvas_df)} rows")
                        elif r.action == 'sort':
                            canvas_title = "Sorted Data"
                        elif r.action == 'filter':
                            canvas_title = "Filtered Data"
                            summary_parts.append(f"Filtered data to {len(canvas_df)} rows")
                        else:
                            canvas_title = f"Result: {r.action}"
                        break
        
        # Fallback to latest file if still no canvas (e.g. just loaded file)
        if canvas_df is None:
            latest_id = session.get_latest_file_id()
            if latest_id:
                canvas_df = session.dataframes.get(latest_id)
                canvas_file_id = latest_id
                canvas_title = "Current Data"
        
        # Custom canvas logic
        canvas = None
        if canvas_df is not None:
            canvas = self._build_canvas(canvas_df, canvas_title, canvas_file_id)
            
        # NEW: Collect generated files for Orchestrator awareness
        generated_files = []
        if canvas_file_id and canvas_file_id in session.file_paths:
             path = session.file_paths[canvas_file_id]
             generated_files.append({
                 "file_id": canvas_file_id,
                 "file_name": canvas_file_id, # often same as ID
                 "file_path": path,
                 "file_type": "spreadsheet"
             })
        
        # Add to combined result
        combined_result['generated_files'] = generated_files
        
        # Construct StandardAgentResponse (v2)
        # Fix: Ensure canvas_data is explicitly exposed for Orchestrator V2 extraction
        standard_response = {
            'status': "success" if all_success else "error",
            'summary': combined_result.get('summary', "Execution completed."),
            'data': combined_result.get('results', []), # Client gets full details
            # 'canvas_display': canvas, # REMOVED: Legacy support removed to prevent conflicts
            'canvas_data': canvas.get('canvas_data') if canvas else None,
            'canvas_type': canvas.get('canvas_type', 'spreadsheet') if canvas else None,
            'canvas_title': canvas.get('canvas_title') if canvas else None,
            'error_message': None if all_success else "; ".join(r.error for r in results if r.error)
        }

        # Legacy nesting for orchestrator compatibility
        combined_result['standard_response'] = standard_response

        return ExecuteResponse(
            status=TaskStatus.COMPLETE if all_success else TaskStatus.ERROR,
            success=all_success,
            result=combined_result,
            canvas_display=canvas,
            standard_response=standard_response,
            error=None if all_success else "; ".join(r.error for r in results if r.error)
        )
    
    async def _step_export(
        self,
        params: Dict,
        df: pd.DataFrame,
        session: Session,
        thread_id: str
    ) -> StepResult:
        """Export data to file."""
        try:
            filename = params.get('filename') or params.get('file_name') or 'export.xlsx'
            file_format = params.get('format') or params.get('file_type') or 'xlsx'
            
            if df is None:
                return StepResult(action='export', success=False, error="No data loaded")
            
            import os
            # Standard root storage
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
            storage_dir = os.path.join(root_dir, 'storage', 'spreadsheet_agent')
            os.makedirs(storage_dir, exist_ok=True)
            
            # Helper to ensure extension
            target_ext = f".{file_format.lstrip('.')}"
            if not filename.lower().endswith(target_ext.lower()):
                filename += target_ext
                
            filepath = os.path.join(storage_dir, filename)
            
            if 'csv' in file_format.lower():
                df.to_csv(filepath, index=False)
            else:
                df.to_excel(filepath, index=False)
                
            # CRITICAL: Register in session state!
            self.state.store_dataframe(thread_id, filename, df, filepath)
            logger.info(f"[_step_export] Registered exported file in session: {filename}")
            
            return StepResult(
                action='export',
                success=True,
                result={
                    'file_path': filepath, 
                    'filename': filename,
                    'file_id': filename, # Use filename as ID for simplicity
                    'generated_files': [{
                        "file_name": filename,
                        "file_path": filepath,
                        "file_type": "spreadsheet",
                        "file_id": filename
                    }] # Inform orchestrator with full metadata
                },
                df_modified=False
            )
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return StepResult(action='export', success=False, error=str(e))

    def _sanitize_for_json(self, obj: Any) -> Any:
        """Ensure object is JSON serializable."""
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, pd.DataFrame):
            return self._sanitize_for_json(obj.head(50).to_dict(orient='records'))
        if isinstance(obj, pd.Series):
             return self._sanitize_for_json(obj.to_dict())
        if hasattr(obj, 'item') and hasattr(obj, 'dtype'): # numpy scalar
             return obj.item()
        if isinstance(obj, dict):
            return {str(k): self._sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
             return [self._sanitize_for_json(v) for v in obj]
        
        # Fallback to string representation for unknown objects
        return str(obj)

    def _build_canvas(
        self,
        df: pd.DataFrame,
        title: str,
        file_id: str = None
    ) -> Dict[str, Any]:
        """Build canvas display for frontend using Central Canvas Service."""
        # Use factory method to ensure V2 compliance
        display = CanvasService.build_spreadsheet_view(
            filename=file_id or "spreadsheet",
            dataframe=df,
            title=title
        )
        
        # Extract the dict representation
        canvas_dict = display.model_dump()
        
        # Inject file_id as extra metadata (legacy support)
        canvas_dict['file_id'] = file_id
        
        return canvas_dict


# ========================================================================
# GLOBAL INSTANCE
# ========================================================================

spreadsheet_agent = SpreadsheetAgent()
