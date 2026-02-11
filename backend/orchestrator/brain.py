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

from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langchain_core.runnables import RunnableConfig

from .schemas import TaskItem, TaskStatus, TaskPriority, PlanPhase, ParallelAction
from .content_orchestrator import get_optimized_llm_context
from backend.services.inference_service import inference_service, InferencePriority

logger = logging.getLogger(__name__)


class BrainDecision(BaseModel):
    """
    The Brain's output - what resource to activate and with what parameters.

    Supports:
    - Direct execution: agent, tool, python, terminal
    - Planning: action_type='plan' with execution_plan
    - Re-planning: action_type='replan' to modify existing plan
    - Parallel: action_type='parallel' with parallel_actions
    - Phase management: phase_complete for LLM-driven phase advancement
    - Human-in-the-loop: requires_approval for sensitive operations
    """

    action_type: str = Field(
        ...,
        description="Type: 'agent'|'tool'|'python'|'terminal'|'plan'|'replan'|'parallel'|'finish'|'skip'",
    )
    resource_id: Optional[str] = Field(
        None, description="Identifier for the resource (agent_id, tool_name, etc.)"
    )
    payload: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Parameters for the action execution"
    )
    reasoning: Optional[str] = Field(None, description="Why this action was chosen")
    user_response: Optional[str] = Field(
        None,
        description="The final answer to the user. MUST be detailed, thorough, and strictly follow length/style constraints (e.g. '100 words'). If the user asked for a story or long text, provide the FULL text here.",
    )
    memory_updates: Optional[Dict[str, Any]] = Field(
        None, description="Key-value pairs to store in persistent memory"
    )
    is_finished: bool = Field(False, description="True if the objective is fully met")

    # --- ADAPTIVE PLANNING ---
    execution_plan: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="For action_type='plan'/'replan': List of phases with phase_id, name, goal, depends_on",
    )

    # --- LLM-DRIVEN PHASE COMPLETION ---
    phase_complete: bool = Field(
        False,
        description="Set to True when the current phase's goal is FULLY achieved. LLM must explicitly decide this.",
    )
    phase_goal_verified: Optional[str] = Field(
        None,
        description="Brief explanation of how the phase goal was met (required when phase_complete=True)",
    )

    # --- PARALLEL EXECUTION ---
    parallel_actions: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="For action_type='parallel': List of actions to execute concurrently",
    )

    # --- HUMAN-IN-THE-LOOP ---
    requires_approval: bool = Field(
        False,
        description="Set True for sensitive/destructive operations: sending emails, deleting files, financial transactions, external API calls with side effects",
    )
    approval_reason: Optional[str] = Field(
        None,
        description="When requires_approval=True: Clear explanation of what will happen and why approval is needed",
    )
    fallback_mode: bool = Field(
        False, description="True if entering fallback due to consecutive failures"
    )


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
        Main reasoning entry point with full context awareness.
        """
        todo_list = state.get("todo_list", [])
        memory = state.get("memory", {})
        insights = state.get("insights", {})
        action_history = state.get("action_history", [])
        iteration_count = state.get("iteration_count", 0)
        failure_count = state.get("failure_count", 0)

        # Initialize state FIRST if this is a new conversation with no tasks
        # This must happen before failure checks to avoid fallback on new conversations
        if not todo_list and state.get("original_prompt"):
            return self._initialize_initial_state(state)

        # Check error conditions AFTER initialization check
        if iteration_count > self.max_iterations:
            return self._force_finish_with_error(state, "Maximum iterations reached")

        if failure_count >= self.max_failures:
            return self._enter_fallback_mode(state, memory, insights)

        # Extract insights from last execution if significant
        updated_insights = self._extract_insights_from_last_action(state, insights)

        decision = await self._make_decision(
            state, config, memory, updated_insights, action_history
        )

        updates = self._apply_decision_to_state(state, decision)

        # Include updated insights if any new ones were extracted
        if updated_insights != insights:
            updates["insights"] = updated_insights

        return updates

    def _build_conversation_history_view(
        self, messages: List[Any], limit: int = 10
    ) -> str:
        """Build a view of the recent conversation history (user & assistant messages)."""
        if not messages:
            return "No conversation history."

        recent_messages = messages[-limit:]
        history_lines = []

        for msg in recent_messages:
            # Handle both object and dict (just in case)
            role = "User"
            content = ""

            if hasattr(msg, "type"):
                role = "User" if msg.type == "human" else "Assistant"
                content = msg.content
            elif isinstance(msg, dict):
                role = "User" if msg.get("type") == "human" else "Assistant"
                content = msg.get("content", "")

            # Simple truncation for very long messages to avoid context overflow
            if len(content) > 500:
                content = content[:500] + "... (truncated)"

            history_lines.append(f"{role}: {content}")

        return "\n".join(history_lines)

    async def _make_decision(
        self,
        state: Dict[str, Any],
        config: Optional[RunnableConfig],
        memory: Dict[str, Any],
        insights: Dict[str, str],
        action_history: List[Dict],
    ) -> BrainDecision:
        """
        Use LLM to decide next action based on state.
        """
        from backend.services.agent_registry_service import agent_registry
        from backend.services.tool_registry_service import tool_registry

        active_agents = agent_registry.list_active_agents()
        active_tools = tool_registry.list_tools()

        # UAP: Include full skill context and standardized agent list for selection
        agent_skills_context = agent_registry.get_all_skills_context()
        active_agents = agent_registry.list_active_agents()

        # Build standardized agent list using centralized registry
        agent_list = "\n".join(
            [
                f"- **{a['name']}** (ID: {a['id']}): {a.get('description', '').split('.')[0]}"
                for a in active_agents
            ]
        )

        agent_list = "\n".join(
            [
                f"- **{a['name']}** (ID: {a['id']}): {a['use_when']}"
                for a in structured_agents
            ]
        )

        agent_list = "\n".join(
            [
                f"- **{a['agent_name']}** (ID: {a['id']}): {a['use_when']}"
                for a in structured_agents
            ]
        )
        tool_list = "\n".join(
            [f"- {t['name']}: {t['description']}" for t in active_tools]
        )

        thread_id = (config or {}).get("configurable", {}).get("thread_id", "default")
        optimized_context = get_optimized_llm_context(state, thread_id)
        history_str = optimized_context.get("context", "No history available.")

        todo_preview = self._build_todo_preview(state.get("todo_list", []))

        # Build FULL action history view (never compressed)
        action_history_str = self._build_action_history_view(action_history)

        # Build conversation history view (NEW)
        conversation_history_str = self._build_conversation_history_view(
            state.get("messages", [])
        )

        # Build insights view (key learnings, never compressed)
        insights_str = self._build_insights_view(insights)

        # Build execution plan view if exists
        plan_str = self._build_execution_plan_view(state)

        # Build uploaded files view (NEW)
        files_str = self._build_uploaded_files_view(state.get("uploaded_files", []))

        prompt = f"""You are the Brain of an intelligent orchestrator. achieve the objective by managing tasks and selecting the best resource for each.

## PERSONA
You are a helpful, intelligent, and expressive AI assistant.
- Your goal is to not only solve tasks but to do so in a way that is clear, engaging, and friendly.
- When explaining comprehensive results, be thorough. When answering simple questions, be concise but polite.
- If the user's request implies a need for creativity or detailed explanation, provide it.
- **IMPORTANT**: If the user asks you to use a specific tool (like Python or an Agent), you MUST use it, even if you know the answer directly.
- **NEVER** complain that "no code was provided". You are an intelligent agent; you must WRITE the code yourself based on the user's objective.

## OBJECTIVE
{state.get("original_prompt", "No objective")}

## MEMORY
{json.dumps(memory, indent=2, default=str) if memory else "Empty"}

## KEY INSIGHTS (preserved learnings - NEVER forget these)
{insights_str}

## UPLOADED FILES (Available for tools/agents)
{files_str}
- **CRITICAL**: You can see that files exist, but you DO NOT have their content yet.
- To read or process these files, you MUST use an appropriate Agent (e.g., `DocumentAgent` for PDFs/Docs) or a Tool (e.g., `read_file` or `Python` code).
- **NEVER** hallucinate or guess the contents of a file if it has not been explicitly read in the Action History.

## EXECUTION PLAN
{plan_str}

## COMPLETE ACTION HISTORY (all actions with results)
{action_history_str}

## CONVERSATION HISTORY (Recent interactions)
{conversation_history_str}

## RECENT CONTEXT (CMS optimized)
{history_str}


## TO-DO LIST
{todo_preview}

## CONSECUTIVE FAILURES: {state.get("failure_count", 0)}

## AVAILABLE RESOURCES
AGENTS (specialized, for complex domain work):
{agent_skills_context or agent_list or "None"}

**QUICK REFERENCE - Use these exact names:**
- Browser Automation Agent (ID: browser_automation_agent) → Web navigation, scraping, forms
- Spreadsheet Agent (ID: spreadsheet_agent) → CSV, Excel, data analysis  
- Mail Agent (ID: mail_agent) → Email, Gmail, messages
- Document Agent (ID: document_agent) → PDF, Word, text documents
- Zoho Books Agent (ID: zoho_books_agent) → Invoicing, accounting, finance

TOOLS (fast, direct functions - PREFER over agents when both qualify):
{tool_list or "None"}

## TOOL USE GUIDELINES
1. **Explicit Requests**: If the user asks to "run code", "use python", "search", or "use [AgentName]", you **MUST** prioritize that action type.
2. **Contextual Logic**: If the objective requires calculation, data processing, or current time/date, use **PYTHON** or **TERMINAL**. Do not guess.
3. **No unnecessary planning**: If the task is simple and can be solved with a single tool/agent call, DO NOT create a complex plan. Just execute.

AGENT: Delegate to a specialized agent.
  - Use action_type='agent', resource_id='<Agent Name>' (use exact names from AVAILABLE RESOURCES)
  - Payload must include:
    - 'prompt': A clear, natural language description of what you want the agent to do.
  - **DO NOT** specify technical `action` endpoints unless absolutely necessary. The agent will plan its own actions.
  
  **Agent Selection Guide:**
  - **Browser Automation Agent**: Navigation, scraping, forms, any website interaction
  - **Spreadsheet Agent**: CSV, Excel, data analysis, tables, charts
  - **Mail Agent**: Email, Gmail, messages, drafts
  - **Document Agent**: PDF, Word, text documents
  - **Zoho Books Agent**: Invoicing, accounting, finance
  
  - Example: {{"action_type": "agent", "resource_id": "Browser Automation Agent", "payload": {{"prompt": "Go to example.com and extract all product prices."}}}}
  - Example: {{"action_type": "agent", "resource_id": "Spreadsheet Agent", "payload": {{"prompt": "Calculate the average revenue per region from this CSV."}}}}
  - Example: {{"action_type": "agent", "resource_id": "Document Agent", "payload": {{"prompt": "Summarize this PDF document."}}}}
  - Example: {{"action_type": "agent", "resource_id": "Mail Agent", "payload": {{"prompt": "Find emails from John about the project."}}}}
  - Example: {{"action_type": "agent", "resource_id": "Zoho Books Agent", "payload": {{"prompt": "Show me all unpaid invoices."}}}}

PYTHON: Execute Python code directly in sandbox.
  - Use action_type='python' when user asks to: run code, calculate, compute, process data, parse, convert, generate.
  - **YOU MUST GENERATE THE CODE**. The user will not provide it.
  - **Output**: The output of your code (stdout) and the value of the variable `result` (if assigned) will be returned.
  - Provide: payload.code (the full, valid Python code to execute)
  - Example: {{"action_type": "python", "payload": {{"code": "print(2 + 2)"}}}}
  - The sandbox has access to: pandas, numpy, json, datetime, re, math, statistics.
  - PREFER PYTHON over agents for quick calculations and data processing.

TERMINAL: Shell commands (ls, cat, grep, etc.).

## ADVANCED ACTION TYPES

### PLAN (for complex multi-phase objectives)
Use action_type='plan' when the objective requires multiple distinct phases.
Example: "Analyze data, compare, create report, email" → Create phases: Data Collection → Analysis → Report → Delivery.
Provide execution_plan as list of phases:
```json
"execution_plan": [
  {{"phase_id": "1", "name": "Data Collection", "goal": "Get Q4 and Q3 data", "depends_on": []}},
  {{"phase_id": "2", "name": "Analysis", "goal": "Compare quarters", "depends_on": ["1"]}}
]
```

### REPLAN (dynamic plan modification)
Use action_type='replan' when:
- A phase fails and you need to adjust the plan
- New information changes requirements
- User provides mid-task input requiring pivot
Provide a NEW execution_plan that replaces the current one. Completed phases remain completed.

### PARALLEL (for independent concurrent tasks)
Use action_type='parallel' when multiple independent actions can run simultaneously.
Example: "Get Q4 data AND Q3 data" → Run both SpreadsheetAgent calls in parallel.
Provide parallel_actions as list:
```json
"parallel_actions": [
  {{"action_type": "agent", "resource_id": "SpreadsheetAgent", "payload": {{"instruction": "Get Q4"}}}},
  {{"action_type": "agent", "resource_id": "SpreadsheetAgent", "payload": {{"instruction": "Get Q3"}}}}
]
```

## PHASE COMPLETION (CRITICAL - LLM DECIDES)
YOU must explicitly decide when a phase is complete. The system does NOT auto-advance phases.

**Set phase_complete=True ONLY when:**
- The current phase's GOAL is FULLY achieved (not just one action succeeded)
- You have ALL the data/results needed for that phase
- Provide phase_goal_verified explaining HOW the goal was met

**Example:**
- Phase goal: "Collect Q4 and Q3 sales data"
- After ONE successful data fetch: phase_complete=False (still need more data)
- After BOTH data fetches complete: phase_complete=True, phase_goal_verified="Retrieved Q4 revenue ($2.1M) and Q3 revenue ($1.8M)"

## HUMAN-IN-THE-LOOP (CRITICAL FOR SAFETY)
Set `requires_approval=True` for ANY action that:
- **Sends communications**: emails, messages, notifications
- **Modifies external state**: deleting files, database writes, API calls with side effects
- **Financial operations**: payments, transfers, invoice creation
- **Irreversible actions**: any action that cannot be easily undone

When `requires_approval=True`:
- Provide clear `approval_reason` explaining WHAT will happen
- Include key details: recipient, amount, file name, etc.
- The action will PAUSE until user approves

**Example:**
```json
{{
  "action_type": "agent",
  "resource_id": "MailAgent",
  "payload": {{"instruction": "Send Q4 report to finance@company.com"}},
  "requires_approval": true,
  "approval_reason": "Will send Q4 sales report email to finance@company.com with 3 attachments (report.pdf, data.xlsx, summary.docx)"
}}
```

## DECISION RULES
1. **Simple tasks**: Execute directly with agent/tool/python/terminal.
2. **Complex multi-phase tasks**: Use 'plan' FIRST to create phases, then execute within phases.
3. **Plan problems?**: Use 'replan' to adjust the plan dynamically.
4. **Independent subtasks**: Use 'parallel' to run them concurrently.
5. **TOOL over AGENT**: Prefer tools when both can handle the task.
6. **Within a phase**: Focus only on the current phase's goal.
7. **Phase done?**: Set phase_complete=True with phase_goal_verified when phase goal is met.
8. **SENSITIVE ACTIONS**: Set requires_approval=True with approval_reason for emails, deletions, payments, etc.
9. Set is_finished=True ONLY when ALL phases complete or objective is met.

## OUTPUT
Return JSON with:
- action_type: 'tool'|'agent'|'python'|'terminal'|'plan'|'replan'|'parallel'|'finish'|'skip'
- resource_id: tool name or agent ID (for direct execution)
- payload: parameters for execution
- execution_plan: list of phases (only when action_type='plan' or 'replan')
- parallel_actions: list of actions (only when action_type='parallel')
- phase_complete: True if current phase goal is FULLY met (default False)
- phase_goal_verified: explanation of how goal was met (when phase_complete=True)
- requires_approval: True for sensitive operations (default False)
- approval_reason: explanation of what will happen (when requires_approval=True)
- reasoning: brief explanation
- user_response: final answer (when is_finished=True)
"""

        try:
            # Log the full prompt for debugging
            logger.debug(f"🧠 Brain Prompt:\n{prompt}")

            decision = await inference_service.generate_structured(
                messages=[HumanMessage(content=prompt)],
                schema=BrainDecision,
                priority=InferencePriority.SPEED,
                temperature=0.5,
            )

            logger.info(f"🧠 Brain Decision: {decision.model_dump_json(indent=2)}")
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

    def _build_action_history_view(
        self, action_history: List[Dict], model_context_window: int = 32000
    ) -> str:
        """
        Build a view of actions taken, with DYNAMIC token budget management.

        LLM-DRIVEN: Token budget adapts based on model context window.
        - Large context (32k+): Allow ~6000 tokens for history
        - Medium context (16k): Allow ~3000 tokens for history
        - Small context (8k): Allow ~1500 tokens for history
        """
        if not action_history:
            return "No actions taken yet."

        # Dynamic token allocation based on model context (15-20% of context for history)
        if model_context_window >= 32000:
            max_tokens = 6000
        elif model_context_window >= 16000:
            max_tokens = 3000
        else:
            max_tokens = 1500

        # Estimate tokens per entry (~50 tokens per entry on average)
        TOKENS_PER_ENTRY = 50
        max_entries = max_tokens // TOKENS_PER_ENTRY

        # If within budget, show all
        if len(action_history) <= max_entries:
            entries_to_show = action_history
            truncated = False
        else:
            # Keep most recent entries (SOTA: recency bias)
            entries_to_show = action_history[-max_entries:]
            truncated = True

        lines = []
        if truncated:
            archived = len(action_history) - max_entries
            lines.append(
                f"[{archived} earlier actions archived | budget: {max_tokens} tokens]"
            )

        for entry in entries_to_show:
            status = "✅" if entry.get("success") else "❌"
            action_type = entry.get("action_type", "?")
            resource = entry.get("resource_id", "")
            result = entry.get("result_summary", "No result")
            iteration = entry.get("iteration", 0)

            lines.append(f"[Step {iteration}] {status} {action_type}:{resource}")
            # Limit result length for token budget
            lines.append(f"   Result: {result[:200]}")

        return "\n".join(lines)

    def _estimate_prompt_tokens(self, prompt: str) -> int:
        """Estimate token count for a prompt (rough: 4 chars ~ 1 token)."""
        return len(prompt) // 4

    def _build_insights_view(self, insights: Dict[str, str]) -> str:
        """Build a view of key insights (never compressed)."""
        if not insights:
            return "No insights yet."

        return "\n".join([f"• {key}: {value}" for key, value in insights.items()])

    def _build_uploaded_files_view(self, uploaded_files: List[Any]) -> str:
        """Build a view of uploaded files available in the context."""
        if not uploaded_files:
            return "No files uploaded."

        lines = []
        for f in uploaded_files:
            # Handle both dict and Pydantic object
            if isinstance(f, dict):
                name = f.get("file_name") or f.get("filename", "Unknown")
                path = f.get("file_path", "Unknown")
                ftype = f.get("file_type", "Unknown")
            else:
                name = getattr(f, "file_name", "Unknown")
                path = getattr(f, "file_path", "Unknown")
                ftype = getattr(f, "file_type", "Unknown")

            lines.append(f"- {name} (Type: {ftype})")
            lines.append(f"  Path: {path}")

        return "\n".join(lines)

    def _build_execution_plan_view(self, state: Dict[str, Any]) -> str:
        """Build a view of the execution plan for complex tasks."""
        execution_plan = state.get("execution_plan")
        current_phase_id = state.get("current_phase_id")

        if not execution_plan:
            return "No plan created yet. For complex multi-phase objectives, use action_type='plan'."

        lines = ["## Plan Phases:"]
        for phase in execution_plan:
            phase_id = phase.get("phase_id", "?")
            name = phase.get("name", "Unnamed")
            goal = phase.get("goal", "")
            status = phase.get("status", "pending")

            # Determine status icon
            if status == "completed":
                icon = "✅"
            elif phase_id == current_phase_id:
                icon = "→"
                status = "IN PROGRESS"
            else:
                icon = "○"

            deps = phase.get("depends_on", [])
            deps_str = f" (after: {', '.join(deps)})" if deps else ""

            lines.append(f"[{icon}] Phase {phase_id}: {name}{deps_str}")
            lines.append(f"    Goal: {goal}")
            if phase.get("result_summary"):
                lines.append(f"    Result: {phase.get('result_summary')[:100]}")

        if current_phase_id:
            lines.append(
                f"\n**Current Phase: {current_phase_id}** - Focus on this phase's goal."
            )

        return "\n".join(lines)

    def _extract_insights_from_last_action(
        self, state: Dict[str, Any], current_insights: Dict[str, str]
    ) -> Dict[str, str]:
        """
        Extract key insights from the last execution result.
        These insights are preserved and never compressed.
        """
        execution_result = state.get("execution_result", {})
        if not execution_result or not execution_result.get("success"):
            return current_insights

        iteration = state.get("iteration_count", 0)
        output = execution_result.get("output")

        if not output:
            return current_insights

        # Create a copy to avoid mutating
        updated_insights = dict(current_insights)

        # Extract insight from significant results
        insight_key = f"step_{iteration}"

        # Extract meaningful data from common result patterns
        if isinstance(output, dict):
            # Look for data/result/message fields
            for key in ["result", "data", "message", "response", "summary"]:
                if key in output and output[key]:
                    val = str(output[key])
                    if len(val) > 20:  # Only significant results
                        updated_insights[insight_key] = val[:200]
                        break
        elif isinstance(output, str) and len(output) > 20:
            updated_insights[insight_key] = output[:200]

        return updated_insights

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
            "todo_list": [initial_task.model_dump()],
            "memory": {},
            "insights": {},
            "action_history": [],
            "execution_plan": None,  # No plan initially
            "current_phase_id": None,
            "iteration_count": 0,
            "failure_count": 0,
            "last_failure_id": None,
            "current_task_id": initial_task.task_id,
            "decision": BrainDecision(
                action_type="skip",
                reasoning="Initializing system state",
            ).model_dump(),
        }

    def _apply_decision_to_state(
        self, state: Dict[str, Any], decision: BrainDecision
    ) -> Dict[str, Any]:
        """Apply the Brain's decision to the state, including plan/replan and parallel handling."""
        todo_list = state.get("todo_list", [])
        memory = state.get("memory", {})
        current_task_id = state.get("current_task_id")
        execution_plan = state.get("execution_plan")
        current_phase_id = state.get("current_phase_id")

        decision_dump = decision.model_dump()
        # Ensure payload is never None (safety for Hands dispatcher)
        if decision_dump.get("payload") is None:
            decision_dump["payload"] = {}

        updates = {
            "decision": decision_dump,
            "iteration_count": state.get("iteration_count", 0) + 1,
        }

        # === HUMAN-IN-THE-LOOP: Check if approval is required ===
        if decision.requires_approval:
            updates["pending_approval"] = True
            updates["pending_decision"] = decision.model_dump()
            logger.info(
                f"⏸️ ACTION REQUIRES APPROVAL: {decision.approval_reason or 'Sensitive operation'}"
            )
            # Return early - don't execute, wait for approval
            return updates

        # Handle PLAN action - store NEW execution plan
        if decision.action_type == "plan" and decision.execution_plan:
            updates["execution_plan"] = decision.execution_plan
            # Set first phase as current
            if decision.execution_plan:
                updates["current_phase_id"] = decision.execution_plan[0].get("phase_id")
            logger.info(
                f"📋 Created execution plan with {len(decision.execution_plan)} phases"
            )

        # Handle REPLAN action - preserve completed phases, replace pending ones
        if decision.action_type == "replan" and decision.execution_plan:
            # === REPLAN VALIDATION ===
            validated_plan, validation_errors = self._validate_execution_plan(
                decision.execution_plan
            )
            if validation_errors:
                logger.warning(f"⚠️ Replan validation issues: {validation_errors}")

            # Preserve completed phases from old plan
            completed_phases = []
            if execution_plan:
                completed_phases = [
                    p for p in execution_plan if p.get("status") == "completed"
                ]

            # Merge: completed phases + validated new plan
            new_plan = completed_phases + validated_plan
            updates["execution_plan"] = new_plan

            # Set current phase to first pending in new plan
            first_pending = next(
                (p for p in new_plan if p.get("status") != "completed"), None
            )
            if first_pending:
                updates["current_phase_id"] = first_pending.get("phase_id")

            logger.info(
                f"🔄 Re-planned: kept {len(completed_phases)} completed, added {len(validated_plan)} new phases"
            )

        # Handle PARALLEL action - just pass through, Hands will execute
        # No special state handling needed here

        # Handle direct execution actions
        if decision.action_type not in ("finish", "skip", "plan", "replan", "parallel"):
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

            updates["current_task_id"] = current_task_id
            updates["todo_list"] = todo_list

        if decision.memory_updates:
            memory.update(decision.memory_updates)
            updates["memory"] = memory

        # === LLM-DRIVEN PHASE COMPLETION ===
        # Only advance phase when LLM explicitly sets phase_complete=True
        if decision.phase_complete and current_phase_id and execution_plan:
            # Find and mark current phase as completed
            updated_plan = list(execution_plan)  # Copy
            for idx, phase in enumerate(updated_plan):
                if phase.get("phase_id") == current_phase_id:
                    updated_plan[idx] = {
                        **phase,
                        "status": "completed",
                        "goal_verified": decision.phase_goal_verified
                        or "LLM verified goal met",
                    }
                    break

            # Find next phase whose dependencies are all satisfied
            completed_phase_ids = {
                p.get("phase_id")
                for p in updated_plan
                if p.get("status") == "completed"
            }

            next_phase_id = None
            for phase in updated_plan:
                if phase.get("status") in ("completed", "skipped"):
                    continue

                deps = phase.get("depends_on", [])
                if all(dep in completed_phase_ids for dep in deps):
                    next_phase_id = phase.get("phase_id")
                    break

            updates["execution_plan"] = updated_plan
            updates["current_phase_id"] = next_phase_id  # None if all done

            logger.info(
                f"✅ Phase '{current_phase_id}' verified complete by LLM → Next: {next_phase_id or 'ALL DONE'}"
            )

        # Handle finish
        is_finished = decision.is_finished or decision.action_type == "finish"
        if is_finished:
            updates["final_response"] = decision.user_response or "Task complete."

            # CRITICAL: Append the final response to the message history so it appears in the chat
            # The 'messages' key in State triggers add_messages reducer which appends to the list
            from langchain_core.messages import AIMessage
            import uuid
            import time

            # Create a fully-formed message with ID and timestamp to ensure frontend acceptance
            final_msg_id = str(uuid.uuid4())
            timestamp = time.time()

            updates["messages"] = [
                AIMessage(
                    content=updates["final_response"],
                    id=final_msg_id,
                    additional_kwargs={"timestamp": timestamp, "id": final_msg_id},
                )
            ]

            updates["current_task_id"] = None

            for task in todo_list:
                if task.get("status") in (TaskStatus.PENDING, TaskStatus.IN_PROGRESS):
                    task["status"] = "completed" if is_finished else "skipped"
            updates["todo_list"] = todo_list

        return updates

    def _validate_execution_plan(
        self, execution_plan: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        """
        Validate execution plan phases are well-formed.

        Returns:
            (validated_plan, errors) - validated plan and list of any validation errors
        """
        errors = []
        validated = []
        phase_ids = set()

        for i, phase in enumerate(execution_plan):
            # Validate required fields
            if not phase.get("phase_id"):
                phase["phase_id"] = f"phase_{i + 1}"
                errors.append(f"Phase {i + 1}: Missing phase_id, auto-assigned")

            if not phase.get("name"):
                phase["name"] = f"Phase {i + 1}"
                errors.append(f"Phase {i + 1}: Missing name, auto-assigned")

            if not phase.get("goal"):
                errors.append(f"Phase {phase.get('phase_id')}: Missing goal")

            # Track phase IDs for dependency validation
            phase_ids.add(phase.get("phase_id"))

            # Ensure status is set to pending for new phases
            if not phase.get("status"):
                phase["status"] = "pending"

            validated.append(phase)

        # Validate dependencies reference existing phase IDs
        for phase in validated:
            deps = phase.get("depends_on", [])
            for dep in deps:
                if dep not in phase_ids:
                    errors.append(
                        f"Phase {phase.get('phase_id')}: Invalid dependency '{dep}'"
                    )

        return validated, errors

    def _enter_fallback_mode(
        self,
        state: Dict[str, Any],
        memory: Dict[str, Any],
        insights: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """Fallback mode: provide best answer from available context."""
        context_summary = {
            "memory": memory,
            "insights": insights or {},
        }
        return {
            "decision": BrainDecision(
                action_type="finish",
                user_response=f"I've encountered multiple issues, but here is what I know: {json.dumps(context_summary, default=str)}",
                is_finished=True,
                fallback_mode=True,
            ).model_dump(),
            "final_response": f"I've encountered multiple issues, but here is what I know: {json.dumps(context_summary, default=str)}",
            "current_task_id": None,
        }

    def _force_finish_with_error(
        self, state: Dict[str, Any], error: str
    ) -> Dict[str, Any]:
        return {
            "final_response": f"Process stopped: {error}",
            "current_task_id": None,
        }
