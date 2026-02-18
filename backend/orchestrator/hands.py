"""
Hands - The Stateless Dispatcher

Executes actions decided by the Brain.
Supports: Agents, Tools, Python code, Terminal commands

Hands is stateless - it takes an action and parameters, executes, returns results.
"""

import logging
import json
import time
import asyncio
from typing import Dict, Any, Optional, List
import httpx

from langchain_core.runnables import RunnableConfig

from .schemas import ActionResult, TaskStatus
from .content_orchestrator import hooks
from .workspace_manager import get_workspace_manager
from backend.utils.retry_utils import RetryManager
from backend.services.agent_registry_service import agent_registry
from backend.services.tool_registry_service import tool_registry
from backend.services.telemetry_service import telemetry_service
from backend.services.code_sandbox_service import code_sandbox
from backend.services.terminal_service import terminal_service
from backend.services.credential_service import get_credentials_for_headers
from database import SessionLocal

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
            result = await self._execute_parallel(parallel_actions, user_id, start_time)
            return self._update_state_with_result(state, result, config)

        # Initialize workspace manager for file tracking
        workspace_manager = get_workspace_manager(thread_id)
        
        # === Direct execution actions ===
        if action_type == "agent":
            result = await self._execute_agent(
                resource_id, payload, user_id, start_time
            )
        elif action_type == "tool":
            result = await self._execute_tool(resource_id, payload, start_time)
        elif action_type == "terminal":
            result = await self._execute_terminal(payload, start_time)
            # Scan for files created by terminal command
            new_files = workspace_manager.scan_for_new_files(created_by="terminal")
            if new_files:
                logger.info(f"📝 Terminal created {len(new_files)} new file(s): {[f.file_name for f in new_files]}")
        elif action_type == "python":
            result = await self._execute_python(payload, start_time, thread_id)
            # Scan for files created by Python code
            new_files = workspace_manager.scan_for_new_files(created_by="python")
            if new_files:
                logger.info(f"📝 Python created {len(new_files)} new file(s): {[f.file_name for f in new_files]}")
        elif action_type == "skip" or action_type == "finish":
            # Return a valid result object even for skip/finish to ensure state consistency
            result = ActionResult(
                action_id=f"{action_type}_{int(time.time())}",
                success=True,
                output={"message": decision_dict.get("user_response")}
                if action_type == "finish"
                else {"skipped": True},
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
        if result.success and result.output:
            processed_output = await hooks.on_task_complete(
                resource_id or action_type,
                {"result": result.output, "status": "completed"},
                thread_id,
            )
            result.output = processed_output

        return self._update_state_with_result(state, result, config)

    async def _execute_parallel(
        self,
        parallel_actions: List[Dict[str, Any]],
        user_id: str,
        start_time: float,
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
                        resource_id, payload, action_start
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
            "payload": payload.get("payload", {}),
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
                agent_name, success, (time.time() - start_time) * 1000
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
                agent_name, False, (time.time() - start_time) * 1000
            )
            
            return ActionResult(
                action_id=f"agent_{agent_id}",
                success=False,
                error_message=f"Agent execution error: {str(e)}",
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    async def _execute_tool(
        self, tool_name: str, payload: Dict[str, Any], start_time: float
    ) -> ActionResult:
        """Execute a tool call."""
        try:
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
            "Terminal", success, (time.time() - start_time) * 1000
        )

        return ActionResult(
            action_id="terminal",
            success=success,
            output=output,
            error_message=output if not success else None,
            execution_time_ms=(time.time() - start_time) * 1000,
        )

    async def _execute_python(
        self, payload: Dict[str, Any], start_time: float, thread_id: str = "default"
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
        
        # Modify code to ensure it saves to workspace directory
        # Add workspace path context at the beginning of code
        modified_code = f"""
import os
os.chdir(r'{workspace_path}')

{code}
"""

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, 
            lambda: code_sandbox.execute_code(modified_code, session_id=session_id)
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

        return ActionResult(
            action_id="python",
            success=success,
            output=output,
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

        # Generate result summary (first 500 chars of string representation)
        result_summary = self._generate_result_summary(result.output)

        # Create action history entry for FULL context awareness
        history_entry = {
            "iteration": iteration,
            "action_type": action_type,
            "resource_id": decision_dict.get("resource_id"),
            "instruction": json.dumps(decision_dict.get("payload", {}))[:200],
            "success": result.success,
            "result_summary": result_summary,
            "result_full": result.output,  # Full result preserved
            "timestamp": time.time(),
            "execution_time_ms": result.execution_time_ms,
        }

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

        # SOTA: Extract Canvas Data from Standard Response (UAP v2)
        # This Bubbles up the canvas display to the top-level state so main.py can see it
        # Handle both dict and Pydantic object responses
        output = result.output
        std_response = None

        if isinstance(output, dict) and "standard_response" in output:
            # Dict format
            std_response = output.get("standard_response")
        elif hasattr(output, "standard_response") and output.standard_response:
            # Pydantic object format (AgentResponse, etc.)
            std_response = output.standard_response

        if std_response:
            # Handle both dict and Pydantic object for std_response
            if isinstance(std_response, dict) and "canvas_display" in std_response:
                canvas = std_response["canvas_display"]
            elif (
                hasattr(std_response, "canvas_display") and std_response.canvas_display
            ):
                canvas = std_response.canvas_display
            else:
                canvas = None

            if canvas:
                logger.info(f"🎨 Hands: Extracting canvas display from agent response")
                updates["has_canvas"] = True
                updates["canvas_type"] = (
                    canvas.get("canvas_type")
                    if isinstance(canvas, dict)
                    else canvas.canvas_type
                )
                updates["canvas_content"] = (
                    canvas.get("canvas_content")
                    if isinstance(canvas, dict)
                    else canvas.canvas_content
                )
                updates["canvas_data"] = (
                    canvas.get("canvas_data")
                    if isinstance(canvas, dict)
                    else canvas.canvas_data
                )
                updates["canvas_title"] = (
                    canvas.get("heading") or canvas.get("canvas_title")
                    if isinstance(canvas, dict)
                    else (canvas.canvas_title or getattr(canvas, "heading", None))
                )

                # For browser view specifically
                canvas_type = updates["canvas_type"]
                if canvas_type == "html":
                    updates["browser_view"] = updates["canvas_content"]
                    updates["current_view"] = "browser"
                elif canvas_type == "plan_graph":
                    updates["plan_view"] = updates["canvas_data"]
                    updates["current_view"] = "plan"
                elif canvas_type == "email_preview":
                    updates["current_view"] = "browser"
                elif canvas_type == "spreadsheet":
                    updates["current_view"] = "spreadsheet"

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
                logger.info(f"🎉 All phases complete!")

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
                    val = str(output[key])[:200]
                    summary_parts.append(f"{key}: {val}")
            if summary_parts:
                return "; ".join(summary_parts)[:max_length]
            return json.dumps(output, default=str)[:max_length]

        return str(output)[:max_length]
