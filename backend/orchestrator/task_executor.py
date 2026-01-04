"""
Unified task execution with tool-first strategy and dynamic agent fallback.

This module implements the TaskExecutor class which provides a unified interface
for executing tasks with the following strategy:
1. Attempt fast tool execution (if parameters are valid)
2. On tool failure, fallback to primary agent
3. On primary agent failure, try fallback agents
4. Track execution metrics and path for monitoring

The executor integrates parameter validation, tool routing, and agent execution
into a single cohesive interface with complete observability.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Optional, Awaitable, List
from dataclasses import dataclass, field
import logging

from orchestrator.parameter_validator import ParameterValidator, ParameterContext
from models import Task

logger = logging.getLogger(__name__)


class ExecutionPath(Enum):
    """Represents the path an execution took through tool/agent attempts."""
    TOOL_SUCCESS = "tool_success"
    TOOL_FAILURE_AGENT_SUCCESS = "tool_failure_agent_success"
    TOOL_FAILURE_AGENT_FALLBACK_SUCCESS = "tool_failure_agent_fallback_success"
    AGENT_SUCCESS = "agent_success"
    AGENT_FALLBACK_SUCCESS = "agent_fallback_success"
    NO_EXECUTION = "no_execution"


@dataclass
class ExecutionMetrics:
    """Metrics collected during task execution."""
    total_time_ms: float
    tool_time_ms: Optional[float] = None
    agent_time_ms: Optional[float] = None
    tool_attempts: int = 0
    agent_attempts: int = 0
    fallback_count: int = 0
    execution_path: ExecutionPath = ExecutionPath.NO_EXECUTION

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_time_ms": self.total_time_ms,
            "tool_time_ms": self.tool_time_ms,
            "agent_time_ms": self.agent_time_ms,
            "tool_attempts": self.tool_attempts,
            "agent_attempts": self.agent_attempts,
            "fallback_count": self.fallback_count,
            "execution_path": self.execution_path.value
        }


@dataclass
class TaskExecution:
    """Result of executing a single task."""
    task_name: str
    task_description: str
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    tool_used: Optional[str] = None
    agent_used: Optional[str] = None
    execution_path: ExecutionPath = ExecutionPath.NO_EXECUTION
    metrics: Optional[ExecutionMetrics] = None
    parameters_used: Dict[str, Any] = field(default_factory=dict)
    timestamp_start: Optional[datetime] = None
    timestamp_end: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            "task_name": self.task_name,
            "task_description": self.task_description,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "tool_used": self.tool_used,
            "agent_used": self.agent_used,
            "execution_path": self.execution_path.value,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "parameters_used": self.parameters_used,
            "timestamp_start": self.timestamp_start.isoformat() if self.timestamp_start else None,
            "timestamp_end": self.timestamp_end.isoformat() if self.timestamp_end else None
        }


class TaskExecutor:
    """
    Executes tasks with tool-first strategy and automatic agent fallback.

    This class provides unified task execution with the following features:
    - Parameter validation before tool execution
    - Fast tool execution path when parameters are valid
    - Automatic fallback to primary agent on tool failure
    - Fallback chain for agent execution
    - Complete execution metrics and path tracking
    - Integrated logging for debugging and monitoring

    The executor is designed to be used within the orchestration graph nodes
    to handle mixed agent/tool execution with proper error handling and metrics.
    """

    def __init__(
        self,
        param_validator: ParameterValidator,
        agent_executor_async: Optional[Callable[[Task, Dict, Optional[Dict]], Awaitable[Dict]]] = None,
        tool_executor_func: Optional[Callable[[str, Dict], Awaitable[Dict]]] = None
    ):
        """
        Initialize the TaskExecutor.

        Args:
            param_validator: ParameterValidator instance for parameter validation
            agent_executor_async: Async function to execute agents
            tool_executor_func: Async function to execute tools
        """
        self.param_validator = param_validator
        self.agent_executor_async = agent_executor_async
        self.tool_executor_func = tool_executor_func or self._default_tool_executor

    async def execute_task(
        self,
        task: Task,
        tool_name: Optional[str],
        intent_params: Dict[str, Any],
        task_agent_pair: Optional[Dict[str, Any]],
        state: Optional[Dict[str, Any]] = None,
        execute_tool_fn: Optional[Callable] = None
    ) -> TaskExecution:
        """
        Execute a single task using tool-first strategy with agent fallback.

        Strategy:
        1. Validate parameters for tool execution
        2. Attempt tool execution if parameters are valid
        3. On tool failure, fallback to primary agent
        4. Try fallback agents if primary agent fails
        5. Return detailed metrics and execution path

        Args:
            task: Task to execute
            tool_name: Name of tool to try first (optional)
            intent_params: Parameters extracted from intent
            task_agent_pair: Dict with 'primary' and 'fallbacks' agents
            state: Current orchestration state
            execute_tool_fn: Custom tool execution function (overrides default)

        Returns:
            TaskExecution with results and metrics
        """
        start_time = datetime.now()
        logger.info(f"🚀 [EXECUTE_TASK] Starting: {task.task_name} | Tool: {tool_name} | Agent: {task_agent_pair}")

        # STEP 1: Validate parameters
        param_context = self.param_validator.validate_and_merge(
            task=task,
            tool_name=tool_name or "unknown",
            intent_params=intent_params or {}
        )

        # STEP 2: Try tool execution (fast path)
        if tool_name and param_context.is_valid:
            tool_result = await self._try_tool_execution(
                task=task,
                tool_name=tool_name,
                param_context=param_context,
                execute_tool_fn=execute_tool_fn
            )

            if tool_result is not None:
                end_time = datetime.now()
                tool_time_ms = (end_time - start_time).total_seconds() * 1000

                logger.info(f"✅ [EXECUTE_TASK] SUCCESS via tool '{tool_name}' in {tool_time_ms:.1f}ms")
                return TaskExecution(
                    task_name=task.task_name,
                    task_description=task.task_description,
                    success=True,
                    result=tool_result,
                    tool_used=tool_name,
                    execution_path=ExecutionPath.TOOL_SUCCESS,
                    metrics=ExecutionMetrics(
                        total_time_ms=tool_time_ms,
                        tool_time_ms=tool_time_ms,
                        agent_time_ms=None,
                        tool_attempts=1,
                        agent_attempts=0,
                        fallback_count=0,
                        execution_path=ExecutionPath.TOOL_SUCCESS
                    ),
                    parameters_used=param_context.merged_params,
                    timestamp_start=start_time,
                    timestamp_end=end_time
                )
            else:
                logger.warning(f"⚠️ [EXECUTE_TASK] Tool '{tool_name}' failed - falling back to agent")

        # STEP 3: Fallback to agent execution
        if task_agent_pair:
            agent_result = await self._try_agent_execution(
                task=task,
                task_agent_pair=task_agent_pair,
                state=state,
                start_time=start_time,
                tool_attempted=bool(tool_name)
            )

            if agent_result is not None:
                end_time = datetime.now()
                agent_time_ms = (end_time - start_time).total_seconds() * 1000

                execution_path = (
                    ExecutionPath.TOOL_FAILURE_AGENT_SUCCESS if tool_name
                    else ExecutionPath.AGENT_SUCCESS
                )

                logger.info(f"✅ [EXECUTE_TASK] SUCCESS via agent '{agent_result.get('agent_name')}' in {agent_time_ms:.1f}ms")
                return TaskExecution(
                    task_name=task.task_name,
                    task_description=task.task_description,
                    success=True,
                    result=agent_result.get('result'),
                    agent_used=agent_result.get('agent_name'),
                    execution_path=execution_path,
                    metrics=ExecutionMetrics(
                        total_time_ms=agent_time_ms,
                        tool_time_ms=None,
                        agent_time_ms=agent_time_ms,
                        tool_attempts=1 if tool_name else 0,
                        agent_attempts=agent_result.get('attempts', 1),
                        fallback_count=1 if tool_name else 0,
                        execution_path=execution_path
                    ),
                    parameters_used=param_context.merged_params,
                    timestamp_start=start_time,
                    timestamp_end=end_time
                )
            else:
                error_msg = agent_result.get('error') if agent_result else "Agent execution failed"
                logger.error(f"❌ [EXECUTE_TASK] All agent attempts failed: {error_msg}")

        # STEP 4: All execution paths exhausted
        end_time = datetime.now()
        total_time_ms = (end_time - start_time).total_seconds() * 1000
        error_msg = f"Task '{task.task_name}' failed - no tools/agents available"

        logger.error(f"❌ [EXECUTE_TASK] FAILURE: {error_msg}")
        return TaskExecution(
            task_name=task.task_name,
            task_description=task.task_description,
            success=False,
            result=None,
            error=error_msg,
            execution_path=ExecutionPath.NO_EXECUTION,
            metrics=ExecutionMetrics(
                total_time_ms=total_time_ms,
                tool_time_ms=None,
                agent_time_ms=None,
                tool_attempts=0,
                agent_attempts=0,
                fallback_count=0,
                execution_path=ExecutionPath.NO_EXECUTION
            ),
            parameters_used={},
            timestamp_start=start_time,
            timestamp_end=end_time
        )

    async def _try_tool_execution(
        self,
        task: Task,
        tool_name: str,
        param_context: ParameterContext,
        execute_tool_fn: Optional[Callable] = None
    ) -> Optional[Any]:
        """
        Try to execute a task via tool.

        Args:
            task: Task to execute
            tool_name: Name of the tool
            param_context: Validated and merged parameters
            execute_tool_fn: Custom tool execution function

        Returns:
            Tool result if successful, None otherwise
        """
        try:
            logger.info(f"🔧 [TOOL] Executing tool '{tool_name}' with params: {param_context.merged_params}")

            # Use provided tool executor or default
            tool_executor = execute_tool_fn or self.tool_executor_func
            result = await tool_executor(tool_name, param_context.merged_params)

            # Check for errors in result
            tool_result = result.get("result") if isinstance(result, dict) else result

            if isinstance(tool_result, dict) and "error" in tool_result:
                logger.warning(f"⚠️ [TOOL] Tool returned error: {tool_result.get('error')}")
                return None

            if result.get("success"):
                logger.info(f"✅ [TOOL] Tool succeeded")
                return tool_result
            else:
                logger.warning(f"⚠️ [TOOL] Tool execution failed: {result.get('error')}")
                return None

        except Exception as e:
            logger.error(f"❌ [TOOL] Tool execution error: {e}", exc_info=True)
            return None

    async def _try_agent_execution(
        self,
        task: Task,
        task_agent_pair: Dict[str, Any],
        state: Optional[Dict[str, Any]],
        start_time: datetime,
        tool_attempted: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Try to execute a task via agent with fallback chain.

        Strategy:
        1. Try primary agent if available
        2. Try each fallback agent in order
        3. Return first successful result
        4. Return None if all attempts fail

        Args:
            task: Task to execute
            task_agent_pair: Dict with 'primary' and 'fallbacks' agents
            state: Current orchestration state
            start_time: Start time of execution
            tool_attempted: Whether a tool was already attempted

        Returns:
            Dict with agent_name, result, attempts if successful, None otherwise
        """
        if not self.agent_executor_async:
            logger.error("❌ [AGENT] No agent executor available")
            return None

        try:
            # Try primary agent
            primary_agent = task_agent_pair.get('primary')
            if primary_agent:
                logger.info(f"🤖 [AGENT] Trying primary agent: {primary_agent.get('name')}")
                result = await self.agent_executor_async(
                    task=task,
                    agent=primary_agent,
                    state=state
                )

                if result and result.get('success'):
                    logger.info(f"✅ [AGENT] Primary agent succeeded")
                    return {
                        'agent_name': primary_agent.get('name'),
                        'result': result.get('result'),
                        'attempts': 1,
                        'success': True
                    }
                else:
                    logger.warning(f"⚠️ [AGENT] Primary agent failed")

            # Try fallback agents
            fallbacks = task_agent_pair.get('fallbacks', [])
            for i, fallback_agent in enumerate(fallbacks, 1):
                logger.info(f"🤖 [AGENT] Trying fallback agent #{i}: {fallback_agent.get('name')}")
                result = await self.agent_executor_async(
                    task=task,
                    agent=fallback_agent,
                    state=state
                )

                if result and result.get('success'):
                    logger.info(f"✅ [AGENT] Fallback agent #{i} succeeded")
                    return {
                        'agent_name': fallback_agent.get('name'),
                        'result': result.get('result'),
                        'attempts': i + 1,
                        'success': True
                    }
                else:
                    logger.warning(f"⚠️ [AGENT] Fallback agent #{i} failed")

            logger.error(f"❌ [AGENT] All agents failed for task '{task.task_name}'")
            return None

        except Exception as e:
            logger.error(f"❌ [AGENT] Agent execution error: {e}", exc_info=True)
            return None

    async def _default_tool_executor(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Default tool executor (should be overridden in practice).

        Args:
            tool_name: Name of the tool to execute
            params: Parameters for the tool

        Returns:
            Dict with success and result/error keys
        """
        logger.warning(f"⚠️ [TOOL] Using default executor for '{tool_name}' - implement custom executor")
        return {"success": False, "error": "No tool executor configured"}
