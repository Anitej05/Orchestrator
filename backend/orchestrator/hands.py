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
from typing import Dict, Any, Optional
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
        """
        decision_dict = state.get("decision")
        if not decision_dict:
            return {"error": "No brain decision found in state"}

        action_type = decision_dict.get("action_type")
        resource_id = decision_dict.get("resource_id")
        payload = decision_dict.get("payload", {})
        
        # Get thread_id and user_id for credential lookup
        owner = (config or {}).get("configurable", {}).get("owner", {})
        user_id = owner.get("user_id", "system")
        thread_id = (config or {}).get("configurable", {}).get("thread_id", "default")

        start_time = time.time()
        logger.info(f"🚀 Hands: Executing {action_type} -> {resource_id}")

        result = None
        
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
        """Update the state with the execution result."""
        updates = {
            "execution_result": result.dict(),
            "error": result.error_message if not result.success else None,
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

        return updates
