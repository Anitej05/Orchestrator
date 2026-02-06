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
from utils.retry_utils import RetryManager
from services.agent_registry_service import agent_registry
from services.tool_registry_service import tool_registry
from services.telemetry_service import telemetry_service
from services.code_sandbox_service import code_sandbox
from services.terminal_service import terminal_service
from services.credential_service import get_credentials_for_headers
from database import SessionLocal

logger = logging.getLogger(__name__)


class Hands:
    """
    The execution dispatcher.
    Takes action decisions from Brain and executes them.
    """

    def __init__(self):
        self.timeout_map = {
            "agent": 60.0,
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
                output={"message": "Execution plan created", "phases": len(decision_dict.get("execution_plan", []))},
                execution_time_ms=(time.time() - start_time) * 1000,
            )
            logger.info(f"📋 Plan created with {len(decision_dict.get('execution_plan', []))} phases")
            return self._update_state_with_result(state, result, config)
        
        # === REPLAN action: acknowledge plan modification ===
        if action_type == "replan":
            new_phases = decision_dict.get("execution_plan", [])
            result = ActionResult(
                action_id=f"replan_{int(time.time())}",
                success=True,
                output={"message": "Execution plan modified", "new_phases": len(new_phases)},
                execution_time_ms=(time.time() - start_time) * 1000,
            )
            logger.info(f"🔄 Plan modified with {len(new_phases)} new phases")
            return self._update_state_with_result(state, result, config)
        
        # === PARALLEL action: execute all actions concurrently ===
        if action_type == "parallel":
            parallel_actions = decision_dict.get("parallel_actions", [])
            result = await self._execute_parallel(parallel_actions, user_id, start_time)
            return self._update_state_with_result(state, result, config)
        
        # === Direct execution actions ===
        if action_type == "agent":
            result = await self._execute_agent(resource_id, payload, user_id, start_time)
        elif action_type == "tool":
            result = await self._execute_tool(resource_id, payload, start_time)
        elif action_type == "terminal":
            result = await self._execute_terminal(payload, start_time)
        elif action_type == "python":
            result = await self._execute_python(payload, start_time)
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
                thread_id
            )
            result.output = processed_output

        return self._update_state_with_result(state, result, config)
    
    async def _execute_parallel(
        self, parallel_actions: List[Dict[str, Any]], user_id: str, start_time: float,
        max_retries: int = 2
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
                    result = await self._execute_agent(resource_id, payload, user_id, action_start)
                elif action_type == "tool":
                    result = await self._execute_tool(resource_id, payload, action_start)
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
                        "retries": retries
                    }
                
                if result.success:
                    return {
                        "index": idx,
                        "action_type": action_type,
                        "resource_id": resource_id,
                        "success": True,
                        "output": result.output,
                        "error": None,
                        "retries": retries
                    }
                
                # Failed - retry with backoff if retries remaining
                if retries < max_retries:
                    backoff = (2 ** retries) * 0.5  # 0.5s, 1s, 2s...
                    logger.info(f"⚠️ Parallel action {idx} failed, retrying in {backoff}s...")
                    await asyncio.sleep(backoff)
                    return await execute_single_with_retry(action, idx, retries + 1)
                
                return {
                    "index": idx,
                    "action_type": action_type,
                    "resource_id": resource_id,
                    "success": False,
                    "output": result.output,
                    "error": result.error_message,
                    "retries": retries
                }
                
            except Exception as e:
                # Exception - retry with backoff if retries remaining
                if retries < max_retries:
                    backoff = (2 ** retries) * 0.5
                    logger.info(f"⚠️ Parallel action {idx} exception, retrying in {backoff}s...")
                    await asyncio.sleep(backoff)
                    return await execute_single_with_retry(action, idx, retries + 1)
                
                return {
                    "index": idx,
                    "action_type": action_type,
                    "resource_id": resource_id,
                    "success": False,
                    "error": str(e),
                    "retries": retries
                }
        
        # Execute all actions concurrently with retry
        tasks = [execute_single_with_retry(action, i) for i, action in enumerate(parallel_actions)]
        results = await asyncio.gather(*tasks)
        
        # Aggregate results
        all_success = all(r.get("success", False) for r in results)
        total_retries = sum(r.get("retries", 0) for r in results)
        combined_output = {
            "parallel_results": results,
            "total_actions": len(parallel_actions),
            "successful": sum(1 for r in results if r.get("success")),
            "failed": sum(1 for r in results if not r.get("success")),
            "total_retries": total_retries
        }
        
        logger.info(f"⚡ Parallel execution: {combined_output['successful']}/{combined_output['total_actions']} succeeded ({total_retries} retries)")
        
        return ActionResult(
            action_id=f"parallel_{int(time.time())}",
            success=all_success,
            output=combined_output,
            error_message=None if all_success else f"{combined_output['failed']} actions failed after retries",
            execution_time_ms=(time.time() - start_time) * 1000,
        )

    async def _execute_agent(
        self, agent_id: str, payload: Dict[str, Any], user_id: str, start_time: float
    ) -> ActionResult:
        """Execute an agent call with robust retry and credential handling."""
        agents = agent_registry.list_active_agents()
        # Case-insensitive match for name or exact match for ID
        agent = next((a for a in agents if a["name"].lower() == agent_id.lower() or a["id"] == agent_id), None)

        if not agent:
            return ActionResult(
                action_id=f"agent_{agent_id}",
                success=False,
                error_message=f"Agent '{agent_id}' not found",
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        instruction = payload.get("instruction", payload.get("prompt", ""))
        
        # Get absolute URL from registry
        db = SessionLocal()
        try:
            base_url = agent_registry.get_agent_url(agent['id'], agent['name'], db)
            auth_headers = get_credentials_for_headers(db, agent['id'], user_id, agent.get('type', 'http_rest'))
        finally:
            db.close()

        url = f"{base_url.rstrip('/')}/process"

        async def _call_agent():
            async with httpx.AsyncClient(timeout=self.timeout_map["agent"]) as client:
                return await client.post(url, json={"request": instruction}, headers=auth_headers)

        try:
            response = await RetryManager.retry_async(
                func=_call_agent,
                max_retries=2,
                operation_name=f"AgentCall({agent['name']})",
                retry_on_status_codes=[502, 503, 504]
            )
            
            agent_result = response.json()
            success = response.status_code == 200
            
            # Deep validation
            if isinstance(agent_result, dict):
                if agent_result.get("success") is False or agent_result.get("status") == "error":
                    success = False

            telemetry_service.log_agent_call(
                agent['name'], success, (time.time() - start_time) * 1000
            )

            return ActionResult(
                action_id=f"agent_{agent['id']}",
                success=success,
                output=agent_result,
                error_message=agent_result.get("error") if not success else None,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            telemetry_service.log_agent_call(
                agent['name'], False, (time.time() - start_time) * 1000
            )

            return ActionResult(
                action_id=f"agent_{agent['id']}",
                success=False,
                error_message=f"Agent connection error: {str(e)}",
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

            return ActionResult(
                action_id=f"tool_{tool_name}",
                success=success,
                output=result_value,
                error_message=exec_result.get("error") or (result_value.get("error") if isinstance(result_value, dict) else None),
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
        self, payload: Dict[str, Any], start_time: float
    ) -> ActionResult:
        """Execute Python code in the sandbox."""
        code = payload.get("code", "")
        session_id = payload.get("session_id", "orchestrator_main")

        if not code:
            return ActionResult(
                action_id="python",
                success=False,
                error_message="No code provided",
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        result = code_sandbox.execute_code(code, session_id=session_id)

        success = result.get("success", False)
        output = result.get("stdout") or result.get("error") or "No output"

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
        
        updates = {
            "execution_result": result.dict(),
            "error": result.error_message if not result.success else None,
            # Append to action history (append_reducer will handle merging)
            "action_history": [history_entry],
        }

        if not result.success:
            failure_count = state.get("failure_count", 0) + 1
            updates["failure_count"] = failure_count
            updates["last_failure_id"] = result.action_id
        else:
            updates["failure_count"] = 0
            updates["last_failure_id"] = None

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
        completed_phases = {p.get("phase_id") for p in execution_plan if p.get("status") == "completed"}
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
            logger.info(f"✅ Phase '{current_phase_id}' complete → Moving to phase '{next_phase_id}'")
        else:
            # All phases complete or no valid next phase
            all_complete = all(
                p.get("status") in ("completed", "skipped") or p.get("phase_id") == current_phase_id
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
        
        if insights != state.get("insights", {}):
            return {"insights": insights}
        return {}
    
    def _generate_result_summary(self, output: Any, max_length: int = 500) -> str:
        """Generate a concise summary of the result for quick reference."""
        if output is None:
            return "No output"
        
        if isinstance(output, str):
            return output[:max_length] + ("..." if len(output) > max_length else "")
        
        if isinstance(output, dict):
            # Extract key fields for summary
            summary_parts = []
            for key in ["result", "message", "data", "response", "output"]:
                if key in output:
                    val = str(output[key])[:200]
                    summary_parts.append(f"{key}: {val}")
            if summary_parts:
                return "; ".join(summary_parts)[:max_length]
            return json.dumps(output, default=str)[:max_length]
        
        return str(output)[:max_length]

