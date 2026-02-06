"""
Brain - The Reasoning Engine

Analyzes state and decides which resources to activate:
- Agent execution
- Tool invocation
- Python code execution
- Terminal commands

The Brain is stateless - it takes the current state and returns a decision.
The Hands node will execute that decision and return a new state.
"""

import logging
import json
import uuid
import time
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from .schemas import TaskItem, TaskStatus, TaskPriority
from .content_orchestrator import get_optimized_llm_context
from services.inference_service import inference_service, InferencePriority

logger = logging.getLogger(__name__)


class BrainDecision(BaseModel):
    """
    The Brain's output - what resource to activate and with what parameters.
    """

    action_type: str = Field(
        ...,
        description="Type of action: 'agent', 'tool', 'python', 'terminal', 'finish', 'skip'",
    )
    resource_id: Optional[str] = Field(
        None, description="Identifier for the resource (agent_id, tool_name, etc.)"
    )
    payload: Dict[str, Any] = Field(
        default_factory=dict, description="Parameters for the action execution"
    )
    reasoning: Optional[str] = Field(None, description="Why this action was chosen")
    user_response: Optional[str] = Field(
        None,
        description="Actual final response text to show the user (required when action_type='finish')",
    )
    memory_updates: Optional[Dict[str, Any]] = Field(
        None, description="Key-value pairs to store in persistent memory"
    )
    is_finished: bool = Field(False, description="True if the objective is fully met")


class Brain:
    """
    The reasoning engine.
    Analyzes the current state and decides the next action.
    """

    def __init__(self):
        self.max_failures = 3
        self.max_iterations = 25

    async def think(
        self, state: Dict[str, Any], config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        """
        Main reasoning entry point.
        """
        todo_list = state.get("todo_list", [])
        memory = state.get("memory", {})
        iteration_count = state.get("iteration_count", 0)
        failure_count = state.get("failure_count", 0)

        if not todo_list and state.get("original_prompt"):
            return self._initialize_initial_state(state)

        if iteration_count > self.max_iterations:
            return self._force_finish_with_error(state, "Maximum iterations reached")

        if failure_count >= self.max_failures:
            return self._enter_fallback_mode(state, memory)

        decision = await self._make_decision(state, config, memory)

        return self._apply_decision_to_state(state, decision)

    async def _make_decision(
        self,
        state: Dict[str, Any],
        config: Optional[RunnableConfig],
        memory: Dict[str, Any],
    ) -> BrainDecision:
        """
        Use LLM to decide next action based on state.
        """
        from services.agent_registry_service import agent_registry
        from services.tool_registry_service import tool_registry

        active_agents = agent_registry.list_active_agents()
        active_tools = tool_registry.list_tools()

        agent_list = "\n".join(
            [f"- {a['name']}: {a['description']}" for a in active_agents]
        )
        tool_list = "\n".join(
            [f"- {t['name']}: {t['description']}" for t in active_tools]
        )

        thread_id = (config or {}).get("configurable", {}).get("thread_id", "default")
        optimized_context = get_optimized_llm_context(state, thread_id)
        history_str = optimized_context.get("context", "No history available.")

        todo_preview = self._build_todo_preview(state.get("todo_list", []))

        prompt = f"""
You are the Brain of an intelligent orchestrator. Your job is to achieve the Current Objective by managing a To-Do list.

CURRENT OBJECTIVE: {state.get("original_prompt", "No objective")}

PERSISTENT MEMORY:
{json.dumps(memory, indent=2, default=str)}

RECENT HISTORY & RESULTS:
{history_str}

FULL TO-DO LIST STATUS:
{todo_preview}

CONSECUTIVE FAILURES: {state.get("failure_count", 0)}

RESOURCES:
AGENTS: {agent_list or "No agents available"}
TOOL: Use 'tool' action_type for registered Python functions from the tool list above. Examples: weather_api, search_engine.
TERMINAL: Use 'terminal' action_type ONLY for shell/CLI commands like 'ls', 'cat file.txt', 'grep pattern file'. Example terminal payload: {{"command": "ls -la"}}.
PYTHON: Use 'python' action_type for executing Python code directly.

DECISION RULES:
1. STRATEGY: If a task fails, try a different approach or resource.
2. FINISHING: Set is_finished=True ONLY when the objective is met.
3. MEANINGFUL RESPONSES: When finishing (is_finished=True), 'user_response' MUST BE THE ACTUAL FINAL RESPONSE. 
   Example: "The stock price of AAPL is $150." or "You're welcome!" 
   DO NOT just describe finishing.
4. GREETINGS: If user greets/thanks you, respond politely and finish immediately.
5. FALLBACK: If failures > 1, provide a direct answer based on available info.

Return JSON matching BrainDecision schema.
"""

        try:
            decision = await inference_service.generate_structured(
                messages=[HumanMessage(content=prompt)],
                schema=BrainDecision,
                priority=InferencePriority.SPEED,
                temperature=0.1,
            )
            return decision
        except Exception as e:
            logger.error(f"Brain LLM failed: {e}")
            return BrainDecision(
                action_type="finish",
                user_response=f"Brain error: {str(e)}",
                is_finished=True,
            )

    def _build_todo_preview(self, todo_list: List[Dict]) -> str:
        if not todo_list:
            return "Empty"
        preview = []
        for t in todo_list:
            status = t.get("status", "pending").upper()
            preview.append(
                f"- [{status}] {t.get('description')} (ID: {t.get('task_id')})"
            )
        return "\n".join(preview)

    def _initialize_initial_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize the todo list with the first task."""
        initial_task = TaskItem(
            task_id=str(uuid.uuid4())[:8],
            description="Initialize objective analysis",
            status=TaskStatus.PENDING,
            priority=10,
        )

        # Return a decision to trigger analysis immediately
        return {
            "todo_list": [initial_task.dict()],
            "memory": {},
            "iteration_count": 0,
            "failure_count": 0,
            "last_failure_id": None,
            "current_task_id": initial_task.task_id,
            "decision": BrainDecision(
                action_type="skip",
                reasoning="Initializing system state",
            ).dict(),
        }

    def _apply_decision_to_state(
        self, state: Dict[str, Any], decision: BrainDecision
    ) -> Dict[str, Any]:
        """Apply the Brain's decision to the state."""
        todo_list = state.get("todo_list", [])
        memory = state.get("memory", {})
        current_task_id = state.get("current_task_id")

        # Determine next task if not already in progress
        if decision.action_type not in ("finish", "skip"):
            # If no task is active, pick the first pending one
            if not current_task_id:
                next_pending = next(
                    (t for t in todo_list if t["status"] == TaskStatus.PENDING), None
                )
                current_task_id = next_pending["task_id"] if next_pending else None
            
            # Mark the current task as in-progress
            if current_task_id:
                for task in todo_list:
                    if task["task_id"] == current_task_id:
                        task["status"] = TaskStatus.IN_PROGRESS
                        break

        # Safety Finish: Force end if no action and not finished
        is_finished = decision.is_finished or decision.action_type == "finish"
        
        updates = {
            "decision": decision.dict(),
            "iteration_count": state.get("iteration_count", 0) + 1,
            "current_task_id": current_task_id,
            "todo_list": todo_list
        }

        if decision.memory_updates:
            memory.update(decision.memory_updates)
            updates["memory"] = memory

        if is_finished:
            updates["final_response"] = decision.user_response or "Task complete."
            updates["current_task_id"] = None

            for task in todo_list:
                if task.get("status") in (TaskStatus.PENDING, TaskStatus.IN_PROGRESS):
                    task["status"] = "completed" if is_finished else "skipped"

        return updates

    def _enter_fallback_mode(
        self, state: Dict[str, Any], memory: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "final_response": f"I've encountered multiple issues, but here is what I know: {json.dumps(memory, default=str)}",
            "current_task_id": None,
        }

    def _force_finish_with_error(
        self, state: Dict[str, Any], error: str
    ) -> Dict[str, Any]:
        return {
            "final_response": f"Process stopped: {error}",
            "current_task_id": None,
        }
