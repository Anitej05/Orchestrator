"""
Hands - The Stateless Dispatcher

Executes actions decided by the Brain.
Supports: Agents, Tools, Python code, Terminal commands

Hands is stateless - it takes an action and parameters, executes, returns results.
"""

import logging
import json
import os
import time
import asyncio
from typing import Dict, Any, Optional, List

from langchain_core.runnables import RunnableConfig

from .schemas import ActionResult, TaskStatus
from .content_orchestrator import hooks
from .workspace_manager import get_workspace_manager
from backend.services.agent_registry_service import agent_registry
from backend.services.tool_registry_service import tool_registry
from backend.services.telemetry_service import telemetry_service
from backend.services.code_sandbox_service import code_sandbox
from backend.services.terminal_service import terminal_service
from services.canvas_service import CanvasService

logger = logging.getLogger(__name__)


class Hands:
    """
    The execution dispatcher.
    Takes action decisions from Brain and executes them.
    """

    def __init__(self):
        self.timeout_map = {
            "agent": 120.0,
            "tool": 30.0,
            "terminal": 30.0,
            "python": 60.0,
        }

    async def execute(
        self, state: Dict[str, Any], config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        """
        Execute the action decided by the Brain.
        Supports: agent, tool, terminal, python, plan, parallel, skip, finish
        """
        decision_dict = state.get("decision")
        if not decision_dict:
            return {"error": "No brain decision found in state"}

        action_type = decision_dict.get("action_type")
        resource_id = decision_dict.get("resource_id")
        payload = decision_dict.get("payload", {})

        # Get thread_id and user_id for credential lookup
        owner = (config or {}).get("configurable", {}).get("owner", {})
        if isinstance(owner, str):
            user_id = owner
        else:
            user_id = owner.get("user_id", "system")
        thread_id = (config or {}).get("configurable", {}).get("thread_id", "default")

        start_time = time.time()
        logger.info(f"🚀 Hands: Executing {action_type} -> {resource_id}")

        result = None

        # === PLAN action: just acknowledge, no execution needed ===
        if action_type == "plan":
            result = ActionResult(
                action_id=f"plan_{int(time.time())}",
                success=True,
                output={
                    "message": "Execution plan created",
                    "phases": len(decision_dict.get("execution_plan", [])),
                },
                execution_time_ms=(time.time() - start_time) * 1000,
            )
            logger.info(
                f"📋 Plan created with {len(decision_dict.get('execution_plan', []))} phases"
            )
            return self._update_state_with_result(state, result, config)

        # === REPLAN action: acknowledge plan modification ===
        if action_type == "replan":
            new_phases = decision_dict.get("execution_plan", [])
            result = ActionResult(
                action_id=f"replan_{int(time.time())}",
                success=True,
                output={
                    "message": "Execution plan modified",
                    "new_phases": len(new_phases),
                },
                execution_time_ms=(time.time() - start_time) * 1000,
            )
            logger.info(f"🔄 Plan modified with {len(new_phases)} new phases")
            return self._update_state_with_result(state, result, config)

        # === PARALLEL action: execute all actions concurrently ===
        if action_type == "parallel":
            parallel_actions = decision_dict.get("parallel_actions", [])
            result = await self._execute_parallel(parallel_actions, user_id, start_time, state)
            return self._update_state_with_result(state, result, config)

        # Initialize workspace manager for file tracking
        workspace_manager = get_workspace_manager(thread_id)
        
        # === Direct execution actions ===
        if action_type == "agent":
            result = await self._execute_agent(
                resource_id, payload, user_id, start_time
            )
        elif action_type == "tool":
            result = await self._execute_tool(resource_id, payload, start_time, state)
        elif action_type == "terminal":
            result = await self._execute_terminal(payload, start_time)
            # Scan for files created by terminal command
            new_files = workspace_manager.scan_for_new_files(created_by="terminal")
            if new_files:
                logger.info(f"📝 Terminal created {len(new_files)} new file(s): {[f.file_name for f in new_files]}")
        elif action_type == "python":
            result = await self._execute_python(payload, start_time, thread_id, state)
            # Scan for files created by Python code
            new_files = workspace_manager.scan_for_new_files(created_by="python")
            if new_files:
                logger.info(f"📝 Python created {len(new_files)} new file(s): {[f.file_name for f in new_files]}")
        elif action_type == "skip" or action_type == "finish":
            # Return a valid result object even for skip/finish to ensure state consistency
            if action_type == "finish":
                user_response = decision_dict.get("user_response", "")
                output_dict = {"message": user_response}
                # Auto-detect canvas from user_response content
                canvas = self._auto_detect_canvas_from_text(user_response)
                if canvas:
                    output_dict["canvas_display"] = canvas
                    logger.info("🎨 Hands: Auto-detected canvas in finish response")
            else:
                output_dict = {"skipped": True}

            result = ActionResult(
                action_id=f"{action_type}_{int(time.time())}",
                success=True,
                output=output_dict,
                execution_time_ms=(time.time() - start_time) * 1000,
            )
            return self._update_state_with_result(state, result, config)
        else:
            result = ActionResult(
                action_id=f"unknown_{action_type}",
                success=False,
                error_message=f"Unknown action type: {action_type}",
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        # Post-process through CMS hooks if success
        # IMPORTANT: Save raw output BEFORE CMS compression replaces it with _content_ref
        raw_output = result.output  # Preserve uncompressed data
        if result.success and result.output:
            processed_output = await hooks.on_task_complete(
                resource_id or action_type,
                {"result": result.output, "status": "completed"},
                thread_id,
            )
            result.output = processed_output
        
        # Attach raw output for downstream use (e.g., Python sandbox injection)
        result._raw_output = raw_output

        return self._update_state_with_result(state, result, config)

    async def _execute_parallel(
        self,
        parallel_actions: List[Dict[str, Any]],
        user_id: str,
        start_time: float,
        state: Dict[str, Any],
        max_retries: int = 2,
    ) -> ActionResult:
        """
        Execute multiple actions concurrently using asyncio.gather.
        LLM-DRIVEN: Includes retry with exponential backoff for failed actions.
        """
        if not parallel_actions:
            return ActionResult(
                action_id="parallel_empty",
                success=False,
                error_message="No parallel actions provided",
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        logger.info(f"⚡ Executing {len(parallel_actions)} actions in parallel")

        async def execute_single_with_retry(
            action: Dict[str, Any], idx: int, retries: int = 0
        ) -> Dict[str, Any]:
            """Execute a single action with retry on failure."""
            action_start = time.time()
            action_type = action.get("action_type")
            resource_id = action.get("resource_id")
            payload = action.get("payload", {})

            try:
                if action_type == "agent":
                    result = await self._execute_agent(
                        resource_id, payload, user_id, action_start
                    )
                elif action_type == "tool":
                    result = await self._execute_tool(
                        resource_id, payload, action_start, state
                    )
                elif action_type == "terminal":
                    result = await self._execute_terminal(payload, action_start)
                elif action_type == "python":
                    result = await self._execute_python(payload, action_start)
                else:
                    return {
                        "index": idx,
                        "action_type": action_type,
                        "resource_id": resource_id,
                        "success": False,
                        "error": f"Unknown action type: {action_type}",
                        "retries": retries,
                    }

                if result.success:
                    return {
                        "index": idx,
                        "action_type": action_type,
                        "resource_id": resource_id,
                        "success": True,
                        "output": result.output,
                        "error": None,
                        "retries": retries,
                    }

                # Failed - retry with backoff if retries remaining
                if retries < max_retries:
                    backoff = (2**retries) * 0.5  # 0.5s, 1s, 2s...
                    logger.info(
                        f"⚠️ Parallel action {idx} failed, retrying in {backoff}s..."
                    )
                    await asyncio.sleep(backoff)
                    return await execute_single_with_retry(action, idx, retries + 1)

                return {
                    "index": idx,
                    "action_type": action_type,
                    "resource_id": resource_id,
                    "success": False,
                    "output": result.output,
                    "error": result.error_message,
                    "retries": retries,
                }

            except Exception as e:
                # Exception - retry with backoff if retries remaining
                if retries < max_retries:
                    backoff = (2**retries) * 0.5
                    logger.info(
                        f"⚠️ Parallel action {idx} exception, retrying in {backoff}s..."
                    )
                    await asyncio.sleep(backoff)
                    return await execute_single_with_retry(action, idx, retries + 1)

                return {
                    "index": idx,
                    "action_type": action_type,
                    "resource_id": resource_id,
                    "success": False,
                    "error": str(e),
                    "retries": retries,
                }

        # Execute all actions concurrently with retry
        tasks = [
            execute_single_with_retry(action, i)
            for i, action in enumerate(parallel_actions)
        ]
        results = await asyncio.gather(*tasks)

        # Aggregate results
        all_success = all(r.get("success", False) for r in results)
        total_retries = sum(r.get("retries", 0) for r in results)
        combined_output = {
            "parallel_results": results,
            "total_actions": len(parallel_actions),
            "successful": sum(1 for r in results if r.get("success")),
            "failed": sum(1 for r in results if not r.get("success")),
            "total_retries": total_retries,
        }

        logger.info(
            f"⚡ Parallel execution: {combined_output['successful']}/{combined_output['total_actions']} succeeded ({total_retries} retries)"
        )

        return ActionResult(
            action_id=f"parallel_{int(time.time())}",
            success=all_success,
            output=combined_output,
            error_message=None
            if all_success
            else f"{combined_output['failed']} actions failed after retries",
            execution_time_ms=(time.time() - start_time) * 1000,
        )

    async def _execute_agent(
        self, agent_id: str, payload: Dict[str, Any], user_id: str, start_time: float
    ) -> ActionResult:
        """
        Execute an agent task with on-demand spawning.
        
        The agent is automatically spawned if not running, executes the task,
        and can be terminated after completion based on policy.
        """
        from backend.services.agent_manager import get_agent_manager
        
        # Get agent info for logging/telemetry
        agent = agent_registry.find_agent(agent_id)
        agent_name = agent.get("name", agent_id) if agent else agent_id
        
        # Prepare task
        instruction = payload.get("instruction", payload.get("prompt", ""))
        task = {
            "prompt": instruction,
            "action": payload.get("action"),
            "payload": payload,
            "task_id": payload.get("task_id"),
            "thread_id": payload.get("thread_id"),
            "user_id": user_id,
        }
        
        try:
            # Get agent manager and ensure it's initialized
            agent_manager = get_agent_manager()
            if not agent_manager._initialized:
                await agent_manager.initialize()
            
            # Execute task (agent spawned automatically if needed)
            logger.info(f"🚀 Executing task on {agent_name} (on-demand)")
            agent_result = await agent_manager.execute(agent_id, task)
            
            # Determine success
            success = agent_result.get("status") != "error"
            if isinstance(agent_result, dict):
                if agent_result.get("success") is False:
                    success = False
            
            # Log telemetry
            telemetry_service.log_agent_call(
                agent_name, success, (time.time() - start_time) * 1000, 
                user_id=user_id, thread_id=payload.get("thread_id", "default")
            )
            
            # Extract error message
            error_message = agent_result.get("error")
            if not error_message and "standard_response" in agent_result:
                error_message = agent_result["standard_response"].get("error_message")
            
            logger.info(f"✅ Agent {agent_name} execution complete (success={success})")
            
            return ActionResult(
                action_id=f"agent_{agent_id}",
                success=success,
                output=agent_result,
                error_message=error_message if not success else None,
                execution_time_ms=(time.time() - start_time) * 1000,
            )
            
        except Exception as e:
            logger.error(f"❌ Agent {agent_name} execution failed: {e}")
            telemetry_service.log_agent_call(
                agent_name, False, (time.time() - start_time) * 1000, 
                user_id=user_id, thread_id=payload.get("thread_id", "default"),
                error_message=str(e)
            )
            
            return ActionResult(
                action_id=f"agent_{agent_id}",
                success=False,
                error_message=f"Agent execution error: {str(e)}",
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    async def _execute_tool(
        self, tool_name: str, payload: Dict[str, Any], start_time: float, state: Dict[str, Any]
    ) -> ActionResult:
        """Execute a tool call."""
        try:
            context = {
                "original_prompt": state.get("original_prompt"),
                "uploaded_files": state.get("uploaded_files", []),
            }
            payload, missing = tool_registry.apply_default_params(
                tool_name, payload, context
            )
            if missing:
                error_msg = f"Missing required tool parameters: {', '.join(missing)}"
                return ActionResult(
                    action_id=f"tool_{tool_name}",
                    success=False,
                    output={"error": error_msg},
                    error_message=error_msg,
                    execution_time_ms=(time.time() - start_time) * 1000,
                )

            exec_result = await tool_registry.execute_tool(tool_name, payload)

            success = exec_result["success"]
            result_value = exec_result.get("result")

            # Check if result itself is an error dict
            if isinstance(result_value, dict) and "error" in result_value:
                success = False

            # Ensure error_message is always a string or None
            error_msg = exec_result.get("error")
            if error_msg is not None and not isinstance(error_msg, str):
                error_msg = str(error_msg)

            if error_msg is None and isinstance(result_value, dict):
                error_msg = result_value.get("error")
                if error_msg is not None and not isinstance(error_msg, str):
                    error_msg = str(error_msg)

            return ActionResult(
                action_id=f"tool_{tool_name}",
                success=success,
                output=result_value,
                error_message=error_msg,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return ActionResult(
                action_id=f"tool_{tool_name}",
                success=False,
                error_message=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    async def _execute_terminal(
        self, payload: Dict[str, Any], start_time: float
    ) -> ActionResult:
        """Execute a terminal command."""
        command = payload.get("command", "")

        if not command:
            return ActionResult(
                action_id="terminal",
                success=False,
                error_message="No command provided",
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        result = terminal_service.execute_command(command)

        success = result["returncode"] == 0
        output = result.get("stdout") or result.get("stderr") or ""

        telemetry_service.log_tool_call(
            "Terminal", success, (time.time() - start_time) * 1000,
            user_id=payload.get("user_id", "system"),
            thread_id=payload.get("thread_id", "default")
        )

        return ActionResult(
            action_id="terminal",
            success=success,
            output=output,
            error_message=output if not success else None,
            execution_time_ms=(time.time() - start_time) * 1000,
        )

    async def _execute_python(
        self, payload: Dict[str, Any], start_time: float, thread_id: str = "default", state: Dict[str, Any] = None
    ) -> ActionResult:
        """Execute Python code in the sandbox, tracking any created files."""
        code = payload.get("code", "")
        session_id = payload.get("session_id", "orchestrator_main")

        if not code:
            return ActionResult(
                action_id="python",
                success=False,
                error_message="No code provided",
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        # Get workspace path for this thread
        workspace_manager = get_workspace_manager(thread_id)
        workspace_path = str(workspace_manager.get_workspace_path())
        
        # === INJECT TOOL RESULTS FROM ACTION HISTORY ===
        # Collect results from previous tool/agent actions so Python code can access them
        tool_results = {}
        if state:
            action_history = state.get("action_history", [])
            for entry in action_history:
                if entry.get("success") and entry.get("action_type") in ("tool", "agent"):
                    resource_id = entry.get("resource_id", "unknown")
                    # Use result_raw (uncompressed) if available, fall back to result_full
                    # result_raw has actual data; result_full may contain _content_ref from CMS
                    result_data = entry.get("result_raw", entry.get("result_full", entry.get("result_summary", "")))
                    
                    # Extract the actual data from the result dict
                    if isinstance(result_data, dict):
                        # Tool results are usually wrapped: {'result': {...actual data...}}
                        actual = result_data.get("result", result_data)
                        tool_results[resource_id] = actual
                    else:
                        tool_results[resource_id] = result_data
                        # Try to parse JSON string
                        if isinstance(result_data, str) and result_data.strip().startswith('{'):
                            try:
                                tool_results[resource_id] = json.loads(result_data)
                            except:
                                pass

                # === UNPACK PARALLEL RESULTS ===
                # Parallel actions wrap multiple tool/agent results in parallel_results list.
                # Extract each sub-result so Python can access them individually via tool_results.
                elif entry.get("action_type") == "parallel":
                    raw = entry.get("result_raw", entry.get("result_full", {}))
                    if isinstance(raw, dict):
                        parallel_results = raw.get("parallel_results", [])
                        for pr in parallel_results:
                            if not isinstance(pr, dict) or not pr.get("success"):
                                continue
                            pr_type = pr.get("action_type", "")
                            pr_resource = pr.get("resource_id", "unknown")
                            pr_output = pr.get("output")
                            if pr_type in ("tool", "agent") and pr_resource and pr_output is not None:
                                # Extract actual data from wrapper
                                if isinstance(pr_output, dict):
                                    actual = pr_output.get("result", pr_output)
                                else:
                                    actual = pr_output
                                # For duplicate tool names (e.g., multiple search_news calls),
                                # collect all results into a list
                                if pr_resource in tool_results:
                                    existing = tool_results[pr_resource]
                                    if isinstance(existing, list):
                                        existing.append(actual)
                                    else:
                                        tool_results[pr_resource] = [existing, actual]
                                else:
                                    tool_results[pr_resource] = actual
            
            # Save tool results as JSON files in workspace for file-based access
            if tool_results:
                for tool_name, data in tool_results.items():
                    safe_name = tool_name.replace('/', '_').replace('\\', '_')
                    filepath = os.path.join(workspace_path, f"{safe_name}_result.json")
                    try:
                        with open(filepath, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=2, default=str)
                    except:
                        pass
        
        # Modify code to ensure it saves to workspace directory
        # Add workspace path context at the beginning of code
        modified_code = f"""
import os
os.chdir(r'{workspace_path}')

{code}
"""

        # Inject tool_results as context variables into the sandbox
        context_vars = {"tool_results": tool_results} if tool_results else None

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, 
            lambda: code_sandbox.execute_code(modified_code, session_id=session_id, context_vars=context_vars)
        )

        success = result.get("success", False)
        stdout = (result.get("stdout") or "").strip()
        result_value = result.get("result")
        error_value = result.get("error")

        if stdout and result_value is not None:
            output = f"{stdout}\nResult: {result_value}"
        elif stdout:
            output = stdout
        elif result_value is not None:
            output = str(result_value)
        elif error_value:
            output = str(error_value)
        else:
            output = "No output"

        output_dict = {"result": output, "code": code}

        # Auto-detect canvas for created files (.xlsx, .csv) in workspace
        try:
            workspace_manager = get_workspace_manager(thread_id)
            workspace_path = workspace_manager.get_workspace_path()
            import glob as glob_mod
            for pattern in ["*.xlsx", "*.csv"]:
                for fpath in glob_mod.glob(str(workspace_path / pattern)):
                    from pathlib import Path
                    p = Path(fpath)
                    if p.suffix == ".csv":
                        import csv
                        import io
                        content = p.read_text(encoding="utf-8", errors="replace")
                        reader = csv.reader(io.StringIO(content))
                        rows_all = list(reader)
                        if rows_all:
                            canvas = CanvasService.build_from_template(
                                "spreadsheet_viewer",
                                {"headers": rows_all[0], "rows": rows_all[1:50], "filename": p.name,
                                 "metadata": {"rows_total": len(rows_all)-1, "rows_shown": min(49, len(rows_all)-1)}},
                                title=p.name,
                            )
                            if canvas:
                                output_dict["canvas_display"] = canvas.model_dump()
                                logger.info(f"🎨 Hands: Auto-canvas for CSV {p.name}")
                                break
                    elif p.suffix == ".xlsx":
                        try:
                            import openpyxl
                            wb = openpyxl.load_workbook(fpath, read_only=True)
                            ws = wb.active
                            rows_all = list(ws.iter_rows(values_only=True))
                            wb.close()
                            if rows_all:
                                headers = [str(h) if h else "" for h in rows_all[0]]
                                data_rows = [[str(c) if c is not None else "" for c in r] for r in rows_all[1:51]]
                                canvas = CanvasService.build_from_template(
                                    "spreadsheet_viewer",
                                    {"headers": headers, "rows": data_rows, "filename": p.name,
                                     "metadata": {"rows_total": len(rows_all)-1, "rows_shown": min(50, len(rows_all)-1)}},
                                    title=p.name,
                                )
                                if canvas:
                                    output_dict["canvas_display"] = canvas.model_dump()
                                    logger.info(f"🎨 Hands: Auto-canvas for XLSX {p.name}")
                                    break
                        except Exception as e:
                            logger.debug(f"Could not read xlsx {fpath}: {e}")
                if "canvas_display" in output_dict:
                    break
        except Exception as e:
            logger.debug(f"Auto-canvas file detection skipped: {e}")

        return ActionResult(
            action_id="python",
            success=success,
            output=output_dict,
            error_message=result.get("error") if not success else None,
            execution_time_ms=(time.time() - start_time) * 1000,
        )

    def _update_state_with_result(
        self,
        state: Dict[str, Any],
        result: ActionResult,
        config: Optional[RunnableConfig] = None,
    ) -> Dict[str, Any]:
        """Update the state with the execution result and record in action history."""
        decision_dict = state.get("decision", {})
        iteration = state.get("iteration_count", 0)
        action_type = decision_dict.get("action_type", "unknown")
        
        # Get thread_id and user_id for workspace managers
        thread_id = (config or {}).get("configurable", {}).get("thread_id", "default")
        user_id = state.get("user_id", "default")

        # Generate result summary from RAW output (before CMS compression)
        # CRITICAL: result.output may be CMS-compressed (_content_ref), which makes
        # the summary opaque to the Brain. Using raw output ensures the Brain can
        # actually see what each action returned, preventing blind repetition loops.
        raw_output = getattr(result, '_raw_output', result.output)
        result_summary = self._generate_result_summary(raw_output)

        # Create action history entry for FULL context awareness
        # For Python actions, extract a meaningful label from the code
        resource_id = decision_dict.get("resource_id")
        instruction = json.dumps(decision_dict.get("payload", {}))[:200]
        if action_type == "python" and not resource_id:
            code = (decision_dict.get("payload") or {}).get("code", "")
            # Extract meaningful label: first comment, or first line of real code
            resource_id = "code"
            for line in code.splitlines():
                line = line.strip()
                if line.startswith("#"):
                    resource_id = line[1:].strip()[:50]
                    break
                elif line and not line.startswith("import ") and not line.startswith("from "):
                    resource_id = line[:50]
                    break
            # Show first 150 chars of actual code as instruction (more useful than JSON)
            instruction = code[:150]
        
        history_entry = {
            "iteration": iteration,
            "action_type": action_type,
            "resource_id": resource_id,
            "instruction": instruction,
            "success": result.success,
            "result_summary": result_summary,
            "result_full": result.output,  # CMS-compressed version for Brain context
            "result_raw": raw_output,  # Uncompressed version for Python sandbox
            "timestamp": time.time(),
            "execution_time_ms": result.execution_time_ms,
        }

        # === ARTIFACT CAPTURE: Record learnings from this execution ===
        try:
            from .artifact_store import get_artifact_store
            artifact_store = get_artifact_store(user_id)
            import asyncio
            # Use fire-and-forget to avoid blocking the main pipeline
            asyncio.ensure_future(artifact_store.capture_from_task(
                action_entry=history_entry,
                state=state,
                objective=state.get("original_prompt", ""),
            ))
        except Exception as e:
            logger.debug(f"Artifact capture skipped: {e}")

        existing_action_history = list(state.get("action_history", []))
        
        # Get workspace managers for file tracking
        workspace_manager = get_workspace_manager(thread_id)
        created_files = workspace_manager.list_files()
        
        # Get shared workspace
        from .shared_workspace import get_shared_workspace_manager
        shared_manager = get_shared_workspace_manager(user_id)
        shared_files = shared_manager.list_files()
        
        updates = {
            "execution_result": result.model_dump(),
            "error": result.error_message if not result.success else None,
            # Append to action history manually
            "action_history": existing_action_history + [history_entry],
            # Track created files and workspace
            "created_files": [f.to_dict() for f in created_files],
            "orchestrator_workspace": str(workspace_manager.get_workspace_path()),
            # Track shared files
            "shared_files": [f.to_dict() for f in shared_files],
            "shared_workspace": str(shared_manager.get_workspace_path()),
        }

        # === CANVAS REGISTRY: Extract and register canvases from agent responses ===
        output = result.output
        canvas = None

        # Path 1: StandardResponse V2 — standard_response.canvas_display
        std_response = None
        if isinstance(output, dict) and "standard_response" in output:
            std_response = output.get("standard_response")
        elif hasattr(output, "standard_response") and output.standard_response:
            std_response = output.standard_response

        if std_response:
            if isinstance(std_response, dict) and "canvas_display" in std_response:
                canvas = std_response["canvas_display"]
            elif (
                hasattr(std_response, "canvas_display") and std_response.canvas_display
            ):
                canvas = std_response.canvas_display

        # Path 2: Direct canvas_display on output dict (Universal Agent, etc.)
        if not canvas and isinstance(output, dict) and "canvas_display" in output:
            canvas = output["canvas_display"]
            if canvas:
                logger.info("🎨 Hands: Found direct canvas_display in result output")

        # Path 3: Nested in output.result dict
        if not canvas and isinstance(output, dict):
            nested_result = output.get("result")
            if isinstance(nested_result, dict) and "canvas_display" in nested_result:
                canvas = nested_result["canvas_display"]
                if canvas:
                    logger.info("🎨 Hands: Found canvas_display in nested result")

        # Path 4: Gmail / email results — auto-build email canvas
        # Detect when any agent returns a list of email messages and wrap as an email canvas
        if not canvas and isinstance(output, dict):
            nested_result = output.get("result")
            if isinstance(nested_result, dict):
                email_messages = nested_result.get("messages")
                if isinstance(email_messages, list) and len(email_messages) > 0:
                    first_msg = email_messages[0] if email_messages else {}
                    if isinstance(first_msg, dict) and any(
                        k in first_msg for k in ("subject", "sender", "from", "snippet", "body")
                    ):
                        canvas = {
                            "canvas_type": "email",
                            "canvas_data": {
                                "messages": email_messages,
                                "total_count": nested_result.get("total_count", len(email_messages)),
                                "query": nested_result.get("query", ""),
                            },
                            "heading": f"Email Results ({len(email_messages)} emails)",
                        }
                        logger.info(f"📧 Hands: Auto-generated email canvas ({len(email_messages)} messages)")

        if canvas:
            logger.info("🎨 Hands: Registering canvas in Canvas Registry")
            c_type = canvas.get("canvas_type") if isinstance(canvas, dict) else canvas.canvas_type
            c_content = canvas.get("canvas_content") if isinstance(canvas, dict) else canvas.canvas_content
            c_data = canvas.get("canvas_data") if isinstance(canvas, dict) else canvas.canvas_data
            c_title = (
                (canvas.get("heading") or canvas.get("canvas_title"))
                if isinstance(canvas, dict)
                else (canvas.canvas_title or getattr(canvas, "heading", None))
            )
            c_confirm = (
                canvas.get("requires_confirmation", False)
                if isinstance(canvas, dict)
                else getattr(canvas, "requires_confirmation", False)
            )
            c_confirm_msg = (
                canvas.get("confirmation_message")
                if isinstance(canvas, dict)
                else getattr(canvas, "confirmation_message", None)
            )
            source_agent = decision_dict.get("resource_id") or action_type
            canvas_id = f"{source_agent}_{c_type}_{int(time.time())}"

            from backend.services.canvas_registry import get_canvas_registry
            registry = get_canvas_registry(thread_id)
            registry.register_sync(
                canvas_id=canvas_id,
                canvas_type=c_type or "unknown",
                source_agent=source_agent,
                canvas_data=c_data,
                canvas_content=c_content,
                canvas_title=c_title,
                requires_confirmation=c_confirm,
                confirmation_message=c_confirm_msg,
            )
            registry_state = registry.get_registry_state()
            updates["canvas_registry"] = registry_state.model_dump()
            updates["active_canvas_id"] = registry.get_active_id()
            compat = registry.get_backward_compat_fields()
            updates.update(compat)

        if not result.success:
            failure_count = state.get("failure_count", 0) + 1
            updates["failure_count"] = failure_count
        else:
            updates["failure_count"] = 0

        current_task_id = state.get("current_task_id")
        todo_list = state.get("todo_list", [])

        if current_task_id:
            for task in todo_list:
                if task.get("task_id") == current_task_id:
                    task["result"] = result.output
                    task["error"] = result.error_message
                    task["status"] = (
                        TaskStatus.COMPLETED if result.success else TaskStatus.FAILED
                    )
                    break

            updates["todo_list"] = todo_list

        # Phase completion is now LLM-DRIVEN via brain.py (phase_complete field)
        # No auto-completion here - the LLM explicitly decides when a phase goal is met

        # === SOTA ENHANCEMENT 2: Parallel Result Insights Extraction ===
        if action_type == "parallel" and result.success:
            insights_updates = self._extract_parallel_insights(state, result)
            updates.update(insights_updates)

        # === EXPLICIT PHASE COMPLETION (LLM-DRIVEN) ===
        if decision_dict.get("phase_complete"):
            phase_updates = self._handle_explicit_phase_completion(state, decision_dict)
            updates.update(phase_updates)

        return updates

    def _handle_explicit_phase_completion(
        self, state: Dict[str, Any], decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle explicit phase completion triggered by the Brain.
        Marks current phase as completed and advances to the next one.
        """
        execution_plan = state.get("execution_plan")
        current_phase_id = state.get("current_phase_id")
        reasoning = decision.get("phase_goal_verified") or "Phase goal met"

        if not execution_plan or not current_phase_id:
            logger.warning(
                "⚠️ Phase completion requested but no active plan/phase found."
            )
            return {}

        # Find current phase index
        current_idx = -1
        current_phase = None
        for idx, phase in enumerate(execution_plan):
            if phase.get("phase_id") == current_phase_id:
                current_phase = phase
                current_idx = idx
                break

        if not current_phase:
            return {}

        logger.info(
            f"✅ Phase '{current_phase.get('name')}' EXPLICITLY completed by Brain."
        )

        # update plan
        new_plan = list(execution_plan)
        new_plan[current_idx] = {
            **current_phase,
            "status": "completed",
            "result_summary": reasoning,
        }

        # Find next phase
        next_phase_id = None
        # Simple logic: find first pending phase that depends on this one, or just next in list if linear
        # But we should respect dependencies.

        # Get set of all completed phases (including this one)
        completed_ids = {
            p.get("phase_id")
            for p in new_plan
            if p.get("status") in ("completed", "skipped")
        }

        for phase in new_plan:
            pid = phase.get("phase_id")
            if pid in completed_ids:
                continue

            # Check dependencies
            deps = phase.get("depends_on", [])
            if not deps or all(d in completed_ids for d in deps):
                next_phase_id = pid
                break

        updates = {"execution_plan": new_plan, "current_phase_id": next_phase_id}

        if next_phase_id:
            logger.info(f"➡ Advancing to phase: {next_phase_id}")
        else:
            logger.info("🎉 All phases completed.")

        return updates

    def _check_phase_completion(
        self, state: Dict[str, Any], result: ActionResult
    ) -> Dict[str, Any]:
        """
        SOTA: Auto-complete current phase when successful action completes.
        Checks if current phase goal is likely achieved and advances to next phase.
        """
        execution_plan = state.get("execution_plan")
        current_phase_id = state.get("current_phase_id")

        if not execution_plan or not current_phase_id or not result.success:
            return {}

        # Find current phase
        current_phase = None
        current_idx = -1
        for idx, phase in enumerate(execution_plan):
            if phase.get("phase_id") == current_phase_id:
                current_phase = phase
                current_idx = idx
                break

        if not current_phase:
            return {}

        # Check if phase should be completed based on result
        # Heuristic: Phase completes after successful action if not already in progress for multiple iterations
        decision = state.get("decision", {})
        action_type = decision.get("action_type")

        # Skip phase advancement for plan/parallel setup actions
        if action_type in ("plan", "skip"):
            return {}

        # Mark current phase as completed with result summary
        updates = {"execution_plan": list(execution_plan)}  # Copy
        updates["execution_plan"][current_idx] = {
            **current_phase,
            "status": "completed",
            "result_summary": self._generate_result_summary(result.output, 200),
        }

        # Find next phase whose dependencies are all completed
        next_phase_id = None
        completed_phases = {
            p.get("phase_id") for p in execution_plan if p.get("status") == "completed"
        }
        completed_phases.add(current_phase_id)  # Include current (just completed)

        for phase in execution_plan:
            if phase.get("phase_id") == current_phase_id:
                continue
            if phase.get("status") in ("completed", "skipped"):
                continue

            # Check if all dependencies are completed
            deps = phase.get("depends_on", [])
            if all(dep in completed_phases for dep in deps):
                next_phase_id = phase.get("phase_id")
                break

        if next_phase_id:
            updates["current_phase_id"] = next_phase_id
            logger.info(
                f"✅ Phase '{current_phase_id}' complete → Moving to phase '{next_phase_id}'"
            )
        else:
            # All phases complete or no valid next phase
            all_complete = all(
                p.get("status") in ("completed", "skipped")
                or p.get("phase_id") == current_phase_id
                for p in execution_plan
            )
            if all_complete:
                updates["current_phase_id"] = None
                logger.info("🎉 All phases complete!")

        return updates

    def _extract_parallel_insights(
        self, state: Dict[str, Any], result: ActionResult
    ) -> Dict[str, Any]:
        """
        SOTA: Extract insights from each parallel action result.
        Maps each parallel result to a numbered insight for future reference.
        """
        insights = dict(state.get("insights", {}))
        iteration = state.get("iteration_count", 0)

        if not result.output or not isinstance(result.output, dict):
            return {}

        parallel_results = result.output.get("parallel_results", [])

        for pr in parallel_results:
            if not pr.get("success"):
                continue

            output = pr.get("output")
            if not output:
                continue

            idx = pr.get("index", 0)
            resource = pr.get("resource_id", "unknown")
            insight_key = f"parallel_{iteration}_{idx}"

            # Extract meaningful content
            if isinstance(output, dict):
                for key in ["result", "data", "message", "response", "summary"]:
                    if key in output and output[key]:
                        val = str(output[key])
                        if len(val) > 20:
                            insights[insight_key] = f"[{resource}] {val[:150]}"
                            break
            elif isinstance(output, str) and len(output) > 20:
                insights[insight_key] = f"[{resource}] {output[:150]}"

        # Always return insights dict if any were extracted or if original state had insights
        if insights or state.get("insights"):
            return {"insights": insights}
        return {}

    def _generate_result_summary(self, output: Any, max_length: int = 500) -> str:
        """Generate a concise summary of the result for quick reference."""
        if output is None:
            return "No output"

        if isinstance(output, str):
            return output[:max_length] + ("..." if len(output) > max_length else "")

        if isinstance(output, dict):
            # UAP v2: Check for StandardAgentResponse summary FIRST
            # This is the dedicated summary specifically for the Orchestrator Brain
            if "standard_response" in output and isinstance(
                output["standard_response"], dict
            ):
                std_summary = output["standard_response"].get("summary")
                if std_summary:
                    return std_summary[:max_length]

            # Extract key fields for summary (Legacy fallback)
            summary_parts = []
            for key in ["result", "message", "data", "response", "output"]:
                if (
                    key in output and key != "standard_response"
                ):  # Avoid recursing into standard_response
                    val = str(output[key])[:400]
                    summary_parts.append(f"{key}: {val}")
            if summary_parts:
                return "; ".join(summary_parts)[:max_length]
            return json.dumps(output, default=str)[:max_length]

        return str(output)[:max_length]

    def _auto_detect_canvas_from_text(self, text: str) -> dict | None:
        """
        Auto-detect structured content in text and return a canvas_display dict.
        
        Detection rules (in priority order):
        1. Code block with 5+ lines → code_viewer canvas
        2. Long markdown (500+ chars) with headers → document_viewer canvas
        
        Returns canvas_display dict or None.
        """
        import re

        if not text or not isinstance(text, str):
            return None

        try:
            # 1. Code block detection — look for ```lang\n...``` with 5+ lines
            code_match = re.search(r'```(\w+)?\n(.*?)```', text, re.DOTALL)
            if code_match:
                lang = code_match.group(1) or 'python'
                code = code_match.group(2).strip()
                if code and code.count('\n') >= 4:
                    canvas = CanvasService.build_from_template(
                        "code_viewer",
                        {"code": code, "language": lang},
                        title=f"Code ({lang})",
                    )
                    if canvas:
                        return canvas.model_dump()

            # 2. Long markdown with structure → document_viewer
            has_headers = bool(re.search(r'^##?\s+', text, re.MULTILINE))
            has_tables = '|---' in text or '| ---' in text
            is_long = len(text) > 500

            if is_long and (has_headers or has_tables):
                # Extract a title from the first header
                title_match = re.search(r'^#\s+(.+)', text, re.MULTILINE)
                title = title_match.group(1).strip() if title_match else "Document"
                canvas = CanvasService.build_from_template(
                    "document_viewer",
                    {"content": text, "title": title, "status": "created"},
                    title=title,
                )
                if canvas:
                    return canvas.model_dump()

        except Exception as e:
            logger.debug(f"Auto-detect canvas failed (non-fatal): {e}")

        return None

