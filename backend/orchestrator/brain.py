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

import os
import logging
import json
import uuid
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from .schemas import TaskItem, TaskStatus
from .content_orchestrator import get_optimized_llm_context
from backend.services.inference_service import inference_service, InferencePriority

logger = logging.getLogger(__name__)


class _PlanPhase(BaseModel):
    """Single phase returned by the dedicated planning LLM call."""
    phase_id: str = Field(..., description="String number: '1', '2', etc.")
    name: str = Field(..., description="Short label (3-6 words)")
    goal: str = Field(..., description="One sentence describing what this step achieves")
    depends_on: List[str] = Field(default_factory=list, description="phase_ids this depends on")


class _PlanningOutput(BaseModel):
    """Structured output from the planning LLM call (wraps list for generate_structured)."""
    phases: List[_PlanPhase] = Field(..., description="List of 2-6 concrete plan phases")


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


class Brain:
    """
    The reasoning engine.
    Analyzes the current state and decides the next action.
    """

    # Generous iteration ceiling -- a pure safety net, not a constraint.
    # A capable model should finish well before this; it only exists to
    # prevent runaway API costs in pathological edge cases.
    _MAX_ITERATIONS = 50

    def __init__(self):
        self.max_iterations = self._MAX_ITERATIONS
        # ContextPipeline: opt-in via env flag for safe migration
        self._use_pipeline = os.getenv("USE_CONTEXT_PIPELINE", "false").lower() == "true"
        self._context_pipeline = None
        if self._use_pipeline:
            try:
                from .context_pipeline import ContextPipeline
                self._context_pipeline = ContextPipeline()
                logger.info("Brain: Using ContextPipeline for prompt assembly")
            except Exception as e:
                logger.warning(f"Brain: ContextPipeline init failed, using legacy prompt: {e}")
                self._use_pipeline = False

    def _compute_max_iterations(self, original_prompt: str) -> int:
        """Return a generous flat iteration budget.

        We intentionally do NOT scale this by task complexity -- the model
        is trusted to finish dynamically when the task is done.  The ceiling
        is just a last-resort safeguard against infinite loops.
        """
        return self._MAX_ITERATIONS

    async def think(
        self, state: Dict[str, Any], config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        """
        Main reasoning entry point with full context awareness.
        """
        # Short-circuit: user just approved a pending action via the REST endpoint.
        # The approved decision is already in state["decision"] with requires_approval=False.
        # Skip all re-planning and return immediately so omni_dispatcher routes to hands.
        # Without this, brain re-runs think() from scratch, picks a different agent, and
        # creates a new approval request → infinite approval loop.
        if state.get("pending_action_approval"):
            logger.info("🧠 Brain: pending_action_approval=True -- skipping re-plan, executing approved decision directly")
            return {"pending_action_approval": False}

        # --- USER INPUT HANDLING ---
        # If the last step requested user input and we haven't received a response yet,
        # we must pause the graph. We return a skip decision so omni_route_condition can pause it.
        if state.get("pending_user_input"):
            if state.get("user_response"):
                # The user has replied! Clear the flag so we can process their response.
                logger.info(f"🧠 Brain: Received user_response: '{state.get('user_response')[:50]}...'. Clearing pending_user_input to resume.")
                # We do not return here; we let the Brain continue thinking with the new context
            else:
                logger.info("🧠 Brain: pending_user_input=True with no response yet -- pausing to wait for user text input")
                decision = BrainDecision(
                    action_type="skip", 
                    reasoning="Waiting for user input"
                )
                return {"decision": decision.model_dump()}

        todo_list = state.get("todo_list", [])
        memory = state.get("memory", {})
        insights = state.get("insights", {})
        action_history = state.get("action_history", [])
        iteration_count = state.get("iteration_count", 0)
        failure_count = state.get("failure_count", 0)

        # Initialize state FIRST if this is a new conversation with no tasks
        # This must happen before failure checks to avoid fallback on new conversations
        if not todo_list and state.get("original_prompt"):
            # Set adaptive iteration budget on first entry
            self.max_iterations = self._compute_max_iterations(
                state.get("original_prompt", "")
            )
            return self._initialize_initial_state(state)

        # MANDATORY PLANNING STEP: If the only task is the sentinel placeholder,
        # run a dedicated planning call to decompose the request before any execution.
        # This is enforced in code (not the prompt) for reliability.
        if (
            len(todo_list) == 1
            and todo_list[0].get("task_id") == "__initial__"
            and state.get("original_prompt")
        ):
            return await self._decompose_into_tasks(state, config)

        # Check iteration limit -- the ONLY hard safety net
        if iteration_count >= self.max_iterations:
            return self._force_finish_with_error(state, "Maximum iterations reached")

        # Code-level consecutive-failure escape: if agents have failed 6+ times
        # in a row, stop re-planning and force finish with a descriptive error.
        # This is a genuine safety net against infrastructure / code bugs
        # (e.g., wrong method signature) that no amount of reprompting will fix.
        if failure_count >= 6:
            recent_agent_failures = [
                e for e in action_history[-failure_count:]
                if not e.get("success") and e.get("action_type") == "agent"
            ]
            if len(recent_agent_failures) >= 6:
                agents = {e.get("resource_id", "unknown") for e in recent_agent_failures}
                last_error = recent_agent_failures[-1].get("result_summary", "unknown error")
                logger.warning(
                    f"🧠 Brain: consecutive-failure escape -- agent(s) {agents} failed "
                    f"{failure_count}x in a row. Forcing finish."
                )
                return self._force_finish_with_error(
                    state,
                    f"Agent(s) {agents} failed {failure_count} consecutive times "
                    f"(last error: {last_error[:200]}). Cannot complete request.",
                )

        # Extract insights from last execution if significant
        updated_insights = self._extract_insights_from_last_action(state, insights)

        decision = None
        try:
            decision = await self._make_decision(
                state, config, memory, updated_insights, action_history
            )
        except Exception as e:
            logger.error(f"🧠 Brain: All inference providers failed: {e}")
            # Force a finish with whatever results are available
            action_history = state.get("action_history", [])
            # Gather any results from previous actions
            collected_results = []
            for entry in action_history:
                if entry.get("success") and entry.get("result_summary"):
                    collected_results.append(entry["result_summary"][:200])
            
            if collected_results:
                summary = "\n\n".join(collected_results)
                decision = {
                    "action_type": "finish",
                    "user_response": f"Here are the results gathered so far:\n\n{summary}\n\n(Note: LLM providers were temporarily unavailable, so processing could not continue.)",
                    "reasoning": f"All inference providers failed: {str(e)[:100]}",
                }
            else:
                # Increment failure count and try again next iteration
                return {
                    "failure_count": failure_count + 1,
                    "iteration_count": iteration_count + 1,
                }
        
        if decision is None:
            return {
                "failure_count": failure_count + 1,
                "iteration_count": iteration_count + 1,
            }
        
        decision = self._apply_stagnation_guard(state, decision)

        # NOTE: The Python-level MANDATORY FINISH override that used to live here
        # has been intentionally removed.  It forcibly overrode the LLM's decision
        # after a single agent success, which was a workaround for weaker models.
        # A capable model should decide on its own when to finish vs. continue.


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
        from backend.services.skill_registry import skill_registry

        active_agents = agent_registry.list_active_agents()
        active_tools = tool_registry.list_tools()

        # SkillRegistry: Compact skill summary (~200 tokens) replaces
        # full skill text dump (~5000+ tokens) for token efficiency.
        # Full skill body is lazy-loaded only when dispatching to a specific agent.
        skill_registry.initialize()
        agent_skills_context = skill_registry.get_skill_summary()

        # === CONTEXT PIPELINE PATH ===
        # When enabled, replaces the monolithic 300-line f-string below
        # with ContextPipeline.assemble() (~4000 tokens vs ~8000+)
        if self._use_pipeline and self._context_pipeline:
            prompt = self._context_pipeline.assemble(state, config)
            logger.debug(
                f"Brain: ContextPipeline prompt ({len(prompt)} chars, "
                f"~{len(prompt) // 4} tokens)"
            )
            return await self._call_llm_and_parse(prompt, state, config)
        active_agents = agent_registry.list_active_agents()

        # Build standardized agent list using centralized registry
        agent_list = "\n".join(
            [
                f"- **{a['name']}** (ID: {a['id']}): {a.get('description', '').split('.')[0]}"
                for a in active_agents
            ]
        )
        tool_list = "\n".join(
            [f"- {t['name']}: {t['description']}" for t in active_tools]
        )

        thread_id = (config or {}).get("configurable", {}).get("thread_id", "default")
        optimized_context = get_optimized_llm_context(state, thread_id)
        history_str = optimized_context.get("context", "No history available.")

        iteration_count = state.get("iteration_count", 0)
        todo_preview = self._build_todo_preview(state.get("todo_list", []))

        # Build FULL action history view (never compressed)
        action_history_str = self._build_action_history_view(action_history)

        # Build LATEST AGENT RESULT block -- shown prominently so Brain can finish immediately
        last_agent_result = state.get("last_agent_result")
        if last_agent_result and last_agent_result.get("success"):
            last_agent_result_str = (
                f"Agent/Tool: **{last_agent_result.get('agent', 'unknown')}** "
                f"(iteration {last_agent_result.get('iteration', '?')})\n"
                f"```\n{last_agent_result.get('result', '')}\n```"
            )
        else:
            last_agent_result_str = None

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

        # === ARTIFACT RETRIEVAL: Pull relevant past experience ===
        artifacts_str = ""
        user_profile_str = "New user - no history."
        try:
            from .artifact_store import get_artifact_store
            user_id = state.get("user_id", "default")
            artifact_store = get_artifact_store(user_id)
            artifacts_str = artifact_store.retrieve_relevant(
                query=state.get("original_prompt", ""),
                top_k=3,
                max_tokens=2000,
            )
            user_profile_str = artifact_store.get_user_profile_prompt()
        except Exception as e:
            logger.debug(f"Artifact retrieval skipped: {e}")

        # === FETCH ACTIVE CONNECTIONS ===
        active_connections_str = "None"
        try:
            from services.integrations.composio_auth import get_auth_manager
            user_id = state.get("user_id", "default")
            auth_mgr = get_auth_manager()
            # This triggers a DB check (and optionally a Composio sync if verified)
            connections = auth_mgr.check_connection_status(user_id)
            if connections and getattr(connections, "get", lambda k: None)("success"):
                connected_apps = connections.get("connected_apps", [])
                if connected_apps:
                    active_connections_str = ", ".join(app if isinstance(app, str) else app.get("app_slug", str(app)) for app in connected_apps)
        except Exception as e:
            logger.warning(f"Failed to fetch active connections for prompt: {e}")

        # Build the last-agent-result block as a separate variable
        # (extracted to avoid nested triple-quoted f-strings which Python 3.11 doesn't support)
        if last_agent_result_str:
            last_result_block = f"""## LATEST AGENT/TOOL RESULT -- MANDATORY FINISH REQUIRED
{last_agent_result_str}

**MANDATORY FINISH -- applies when the task is SINGLE-STEP or ALL steps are done:**
The agent returned success. If the request is now fully satisfied, you MUST call action_type=finish RIGHT NOW.

Rules that CANNOT be overridden:
1. task_summary IS the answer. Even if extracted_data is empty or canvas_display is null, the task_summary text contains what the agent accomplished. Present it to the user.
2. Do NOT call the same agent again for the same purpose. You already have the answer for this step.
3. Do NOT run Python to re-read the spreadsheet, load JSON files, or verify the numbers. Trust the agent success status.
4. Do NOT check if data looks right. Trust the agent success status.
5. Do NOT read spreadsheet_agent_result.json or any _result.json file. The answer is already in your action history above -- use it directly.
6. Phase tasks are planning artifacts, not user-specified steps. When the agent successfully answers the original question, ALL phase tasks are done. Call action_type=finish immediately.

How to finish: Call action_type=finish with user_response = the task_summary text formatted nicely for the user.
"""
        else:
            last_result_block = ""

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
- To read or process these files, you MUST use an appropriate Agent (e.g., `DocumentAgent` for PDFs/Docs) or a Tool.
- **NEVER** hallucinate or guess the contents of a file if it has not been explicitly read in the Action History.
- **AGENT-FIRST RULE**: If uploaded files are present AND a matching agent exists (spreadsheet → SpreadsheetAgent, PDF/doc → DocumentAgent), your **VERY FIRST action** MUST be to call that agent. DO NOT run Python to explore the filesystem before calling the agent -- the agent reads the file directly using its path.
- **COMPLETE QUESTION REQUIRED**: When calling an agent for an uploaded file, include the user's COMPLETE question with ALL filtering, grouping, aggregation, and analysis needed -- in a SINGLE call. **DO NOT** call the agent just to get column headers, schema, or a sample first. A schema-only probe is a wasted iteration. Ask the full question directly.
- **NO PYTHON AFTER AGENT SUCCESS**: If a SpreadsheetAgent or DocumentAgent call succeeded, **DO NOT** run Python to re-read the spreadsheet, filter data, or "verify" the answer. The agent already did the analysis. Read the `task_summary` from action history and call finish.
- **DO NOT** use Python `os.walk`, `os.listdir`, `glob`, or file-reading code to inspect uploaded files. The agent handles that. Python file exploration BEFORE calling an agent wastes iterations and provides no useful information.

## YOUR FILES

### Thread Workspace (This conversation only)
{self._build_created_files_view(state.get("created_files", []), state.get("orchestrator_workspace", "Unknown"))}
- These files are private to THIS conversation
- Use these for temporary analysis, charts, downloads
- They won't be visible in other conversations

### Shared Workspace (All your conversations)
{self._build_shared_workspace_view(state.get("shared_files", []), state.get("shared_workspace", "Unknown"))}
- These files persist ACROSS all your conversations
- Use these for templates, saved reports, reusable data
- When user says "save this for later" or "remember this", put files here
- Files are shared across ALL your conversation threads

## AGENT WORKSPACES (Files created by agents)
{self._build_agent_workspaces_view()}
- These are files stored in agent-specific directories
- You can access these if needed for the task

## EXECUTION PLAN
{plan_str}

## COMPLETE ACTION HISTORY (all actions with results)
{action_history_str}

## SELF-CHECK: REVIEW BEFORE EVERY DECISION
**Before choosing your next action, reflect on your progress:**

1. **Review your action history above.** What have you already accomplished? What data do you already have?
2. **Avoid redundancy.** If a tool already returned the data you need, use that data -- don't call it again with a similar query.
3. **Progress toward the goal.** Each action should bring you meaningfully closer to completing the user's request. If you have all the information needed, move to the next step (e.g., from research → Python → finish).
4. **Know when to finish.** Once you've gathered the data, processed it, and created any requested outputs (CSV, analysis, etc.), present the results to the user with action_type='finish'.
5. **Handle errors gracefully.** If a tool or agent fails, try a different approach or work with what you have. Don't repeat the same failing action.
6. **Be efficient.** The user values results, not exhaustive retries. One good search is better than five similar ones.
7. **NEVER repeat the same action.** If you already called python to read a file, DO NOT call python again to read the same file. The result is in your action history above.
8. **Iteration awareness.** You are on iteration {iteration_count}/{self.max_iterations}. Use your judgement on when to finish -- you have ample room.
9. **MANDATORY PLANNING STEP -- do this before anything else.**
   If the TO-DO LIST above contains exactly ONE task with `task_id = '__initial__'`, you MUST respond with `action_type='plan'`. This is NOT optional and has NO exceptions.
   - Break the user's objective into **2 to 6 concrete, specific steps** -- each one a real, named action (e.g. "Fetch Q4 supplier data from spreadsheet", "Calculate total spend per supplier", "Identify top 5 by volume", "Write summary report").
   - Steps must be specific enough that a person reading the list knows exactly what will happen.
   - Do NOT use vague steps like "Gather data", "Process information", or "Analyse results" -- name the actual operation.
   - Do NOT jump straight to execution. This planning step runs in under 1 second and gives the user a clear preview of exactly what the agent will do.
   - Format: `action_type='plan'`, `execution_plan=[{{"phase_id":"1","name":"Short label","goal":"What this step achieves","depends_on":[]}},...`]

10. **AGENT-FIRST for uploaded files.** If uploaded files are listed above AND the action history is empty (iteration 1), your FIRST action MUST be calling the appropriate agent -- NOT Python file exploration. The agent reads the file using its full path. **Include the user's COMPLETE question with ALL filtering/grouping/analysis needed -- NOT just a schema/header probe. One agent call should answer the full question.**
11. **Agent results stay in action history.** After an agent completes successfully, its result is in your action history above. DO NOT write Python to load JSON files to retrieve agent output. DO NOT run Python to filter the spreadsheet or "verify" the answer. Read the `task_summary` from action history and call `action_type='finish'` immediately.
12. **PROHIBITION on Python after agent success.** If the action history contains a successful SpreadsheetAgent or DocumentAgent result, you are FORBIDDEN from using `action_type='python'` on the next step. The Python filter will FAIL (no Excel header auto-detection), and you already have the answer in action history.

## CONVERSATION HISTORY (Recent interactions)
{conversation_history_str}

## RECENT CONTEXT (CMS optimized)
{history_str}


## TO-DO LIST
{todo_preview}

## CONSECUTIVE FAILURES: {state.get("failure_count", 0)}
{self._build_failure_guidance(state)}

## ACTIVE CONNECTIONS
The user has actively authenticated and connected the following apps: {active_connections_str}

## AUTH-REQUIRED RESPONSES
If the last execution result contains `"status": "pending_auth"` or `"auth_required": true`:
1. Check the ACTIVE CONNECTIONS list above. 
2. If the required app (e.g. gmail) IS in the active connections list, the user HAS just authenticated. **DO NOT** finish with an auth link! Instead, RETRY the agent call immediately since the connection is now ready.
3. If the required app is NOT in the active connections list, you MUST set `action_type = "finish"` immediately. Set `user_response` to the `"message"` field from the auth result — it contains a clickable auth link. Do NOT retry the agent.

## RELEVANT EXPERIENCE (from past interactions)
{artifacts_str or 'No relevant past experience.'}

## USER PROFILE
{user_profile_str}

{last_result_block}
## RESOURCE SELECTION -- PICK THE RIGHT ACTION TYPE

**Match the current step to the RIGHT resource.** Each action type has a specific purpose -- read the agent skills, tool descriptions, and Python scope below to decide which one fits.

**CRITICAL RULES:**
- Code in user_response is **NOT EXECUTED**. Only action_type='python' executes code.
- Files are **NOT CREATED** by finish. Only action_type='python' or 'terminal' can create files.


### 1. AGENT -- Refer to agent skills below for when to use each agent
→ action_type='agent', resource_id='<Agent Name>', payload={{"prompt": "..."}}

**Available agents (see their full capabilities, uses, and limitations below):**
{agent_skills_context or agent_list or '   None'}

### 2. TOOL -- External data APIs
→ action_type='tool', resource_id='tool_name', payload={{...}}
Available tools:
{tool_list or '   None'}

### 3. PYTHON -- Computation, data processing, file generation ONLY
→ action_type='python', payload={{"code": "your_python_code_here"}}

**USE Python for:**
✅ Calculations, data analysis, charting (matplotlib, pandas)
✅ Creating CSV, JSON, Excel files from data you already have
✅ Processing results from tools/agents into deliverables
✅ Parsing structured data, JSON manipulation, text processing

**DO NOT use Python for (use agents instead):**
❌ Navigating websites, web scraping, or extracting data from URLs → Browser Agent
❌ Creating Word/PDF documents → Document Agent
❌ Analyzing user-uploaded spreadsheets → Spreadsheet Agent
❌ Writing/editing actual project code → Coding Agent
❌ Sending/reading emails → Gmail Agent

⚠️ CONNECTION VERIFICATION RULE: If an agent action already SUCCEEDED in the current session, the OAuth connection is PROVEN WORKING. Do NOT call integrations_agent afterwards to "check if [app] is connected" -- that call is redundant and wastes iterations. Rule: if gmail_agent returned emails or confirmed sending → Gmail IS connected → proceed directly to finish. This rule applies to every agent: a successful response = connection confirmed.

**Sandbox details:**
- Modules: pandas, numpy, json, datetime, re, math, statistics, csv, os, requests
- Agent and tool results are in your **ACTION HISTORY** above -- read them there directly. DO NOT write Python code to load `<agent>_result.json` files to retrieve agent results; those files may not exist or may contain a raw dump that is already in your history.
- Files created in the sandbox persist and are tracked automatically
- HTTP calls use proper browser headers (no 403 issues)

### 4. TERMINAL -- Shell commands
→ action_type='terminal', payload={{"command": "..."}}
⚠️ This is a WINDOWS system. Use `dir` not `ls`, `type` not `cat`, `findstr` not `grep`

### 5. FINISH -- All work is done
→ action_type='finish', user_response='your comprehensive answer'
⚠️ ONLY use finish when ALL subtasks are done
⚠️ If user asked to "create a file" and you haven't created it yet → DO NOT finish
✅ Use finish to summarize results AFTER tools/python/agents have done the work

## MULTI-STEP TASK RULES
- If task has multiple parts (e.g., "search, then code, then display"):
  → Execute each part in order: tool → python → finish
  → Do NOT skip ahead to finish after just the first part
- If task needs multiple independent steps → Use action_type='parallel'
- If task needs sequential phases → Use action_type='plan'

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
  "resource_id": "Gmail Agent",
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

## TASK COMPLETION DETECTION - CRITICAL

**You MUST stop when the task is complete. Do NOT continue indefinitely.**

**Ask yourself these questions BEFORE every decision:**

1. **Have I answered the user's original question?** 
   - If YES → Use action_type='finish' immediately
   - If NO → Continue with next action

2. **Is the information I have sufficient to satisfy the objective?**
   - For data/analysis tasks: Do I have the key results/data the user asked for?
   - For research tasks: Have I found the core information requested?
   - For coding tasks: Does the code run and produce the expected output?

3. **Am I making progress or just repeating similar actions?**
   - If the last 2-3 actions gave similar results → You likely have enough data
   - If you're searching for "just one more" piece of info → You probably have enough

**COMPLETION EXAMPLES:**

✅ **User**: "Get Tesla stock news and prices, then predict future"
- After: News fetched + Prices fetched + Python prediction ran with output
- **ACTION**: action_type='finish' with the prediction results

✅ **User**: "Summarize this PDF"
- After: DocumentAgent reads PDF and returns summary
- **ACTION**: action_type='finish' with the summary

✅ **User**: "Calculate Q4 revenue"
- After: Python code runs and returns the calculated value
- **ACTION**: action_type='finish' with the answer

❌ **DON'T**: Keep searching for "more context" or "verification" when you already have a clear answer
❌ **DON'T**: Create additional analysis, charts, or reports unless explicitly requested
❌ **DON'T**: Ask "Would you like me to..." - just finish if the core task is done

## STOPPING/TERMINATION - CRITICAL
- When the objective is COMPLETELY MET and you have the final answer → **MUST use action_type='finish'**
- When task is complete, you MUST use action_type='finish'
- If you have the answer ready, just set action_type='finish' and provide user_response
- **BEFORE calling another tool/agent, ask: "Do I already have what the user needs?"**

## FILE SHARING - WHEN TO USE WHICH WORKSPACE

**Thread Workspace (Private):**
- Use for: Temporary files, analysis results, charts, downloads
- User says: "Create a chart", "Analyze this data", "Download this file"
- Files stay private to this conversation
- **Default location** for all created files

**Shared Workspace (Persistent):**
- Use for: Templates, saved reports, reusable data, important files
- User says: "Save this for later", "Remember this", "Keep this for next time"
- Files available in ALL your conversations
- To share a file: Use Python to copy from thread workspace to shared workspace
  ```python
  import shutil
  shutil.copy('chart.png', '../shared/user_123/chart.png')
  ```

**Example:**
- User: "Create a sales report" → Save to thread workspace
- User: "Save this report for future reference" → Copy to shared workspace
- User: "Use that template from last week" → Read from shared workspace

## INTELLIGENT FILE SHARING - YOU DECIDE

As the orchestrator, YOU decide when to share files based on user intent:

**Automatically SHARE to persistent storage when:**
1. User explicitly says: "save this", "keep this", "remember this"
2. Creating a template (email template, report template, etc.)
3. User marks something as "important" or "permanent"
4. User wants to "reuse" or "use again"
5. File type suggests persistence (.docx report, .json config, etc.)

**Keep PRIVATE (thread workspace) when:**
1. Temporary analysis or charts
2. One-time downloads
3. Scratch/experimental files
4. User doesn't indicate importance
5. File type suggests temporary (.tmp, .cache, .log)

**How to share after creating a file:**
Use Python to copy the file:
```python
import shutil
# After creating a file, copy to shared workspace
shutil.copy('myfile.png', '../shared/test_user/myfile.png')
```

Then tell user: "I've saved this to your persistent storage so it's available in all your conversations."

**Note:** Agents don't have direct access to shared storage - only YOU do. When an agent needs a shared file, YOU provide it to them.

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
- user_response: final answer (when action_type='finish')
"""

        return await self._call_llm_and_parse(prompt, state, config)

    async def _call_llm_and_parse(
        self,
        prompt: str,
        state: Dict[str, Any],
        config: Optional[RunnableConfig] = None,
    ) -> BrainDecision:
        """
        Call the LLM with the assembled prompt and parse response into BrainDecision.

        Shared by both the legacy monolithic prompt path and the ContextPipeline path.
        Also prepends ORBIMESH.md system context if available.
        """
        thread_id = (config or {}).get("configurable", {}).get("thread_id", "default")

        # Prepend ORBIMESH.md system context if available
        try:
            from backend.services.memory_service import get_memory_service
            sys_ctx = get_memory_service(state.get("user_id", "default")).get_system_context()
            if sys_ctx:
                prompt = f"## SYSTEM INSTRUCTIONS (from ORBIMESH.md)\n{sys_ctx}\n\n{prompt}"
        except Exception:
            pass

        try:
            logger.debug(f"Brain Prompt ({len(prompt)} chars):\n{prompt[:500]}...")

            decision = await inference_service.generate_structured(
                messages=[HumanMessage(content=prompt)],
                schema=BrainDecision,
                priority=InferencePriority.SPEED,
                temperature=0.5,
                telemetry_metadata={
                    "thread_id": thread_id,
                    "user_id": state.get("user_id") or state.get("owner_id"),
                    "agent_name": "Brain",
                    "operation_type": "brain_decision",
                },
            )

            logger.info(f"Brain Decision: {decision.model_dump_json(indent=2)}")
            return decision
        except Exception as e:
            logger.error(f"Brain LLM failed: {e}", exc_info=True)
            return BrainDecision(
                action_type="finish",
                user_response=f"Brain error: {str(e)}",
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

        last_entry_idx = len(entries_to_show) - 1
        for idx, entry in enumerate(entries_to_show):
            status = "✅" if entry.get("success") else "❌"
            action_type = entry.get("action_type", "?")
            resource = entry.get("resource_id") or ""
            result = entry.get("result_summary", "No result")
            iteration = entry.get("iteration", 0)
            instruction = entry.get("instruction", "")

            # Show action type with resource (avoid ":None" or ":" suffix)
            label = f"{action_type}:{resource}" if resource else action_type
            lines.append(f"[Step {iteration}] {status} {label}")
            # Show what was requested (helps Brain detect its own repetition)
            if instruction:
                lines.append(f"   Code/Instruction: {instruction[:150]}")
            # Last successful agent/tool entry: show more so Brain can finish from it
            is_last = idx == last_entry_idx
            if is_last and entry.get("success") and action_type in ("agent", "tool"):
                max_result_len = 1200
            elif action_type in ("tool", "python", "agent"):
                max_result_len = 400
            else:
                max_result_len = 200
            lines.append(f"   Result: {result[:max_result_len]}")

        # === REDUNDANCY DETECTION ===
        # Count repeated action patterns to warn the Brain
        from collections import Counter
        action_patterns = []
        for entry in action_history:
            a_type = entry.get("action_type", "")
            a_resource = entry.get("resource_id", "")
            action_patterns.append(f"{a_type}:{a_resource}")
        
        pattern_counts = Counter(action_patterns)
        repeated = {k: v for k, v in pattern_counts.items() if v >= 3}
        if repeated:
            lines.append("")
            lines.append("⚠️ REDUNDANCY WARNING: You are repeating actions!")
            for pattern, count in repeated.items():
                lines.append(f"   → {pattern} called {count} times. STOP repeating this.")
            lines.append("   → Use the data you already have, or call 'finish' now.")

        return "\n".join(lines)

    def _extract_text_output(self, output: Any) -> str:
        """Extract readable text from execution output."""
        if output is None:
            return ""
        if isinstance(output, str):
            text = output.strip()
            if "\nResult:" in text:
                return text.split("\nResult:", 1)[-1].strip()
            return text
        if isinstance(output, dict):
            for key in ("result", "output", "message", "response", "data"):
                value = output.get(key)
                if value:
                    return str(value).strip()
        return str(output).strip()

    def _normalize_signature(self, value: str) -> str:
        """Normalize signature strings to compare repeated actions robustly.

        Collapses all whitespace and lowercases so that capitalisation and minor
        formatting differences (extra spaces, newlines) don't fool the guard.

        WHY: The LLM sometimes paraphrases the same instruction slightly differently
        on each iteration (e.g. different whitespace, capitalised first word). Without
        normalisation these would hash to different signatures and the stagnation guard
        would never fire, allowing infinite loops on tasks like "keep checking the file".
        """
        import re as _re
        return _re.sub(r'\s+', ' ', str(value or "").lower()).strip()

    def _extract_primary_content(self, payload_dict: Dict[str, Any]) -> str:
        """Pull the most meaningful text field from a payload for signature comparison.

        Prefers semantic content (prompt/code/command) over raw JSON so that
        superficially different but semantically identical payloads hash the same.
        Falls back to sorted JSON when no primary field is present.
        """
        content = (
            payload_dict.get("prompt") or
            payload_dict.get("instruction") or
            payload_dict.get("code") or
            payload_dict.get("command") or
            ""
        )
        if content:
            return str(content)
        try:
            return json.dumps(payload_dict, sort_keys=True, separators=(",", ":"), default=str)
        except Exception:
            return str(payload_dict)

    def _build_decision_signature(self, decision: BrainDecision) -> str:
        """Build a stable signature for a proposed decision."""
        payload = decision.payload or {}
        content = self._extract_primary_content(payload)
        return self._normalize_signature(
            f"{decision.action_type}|{decision.resource_id or ''}|{content}"
        )

    def _build_history_signature(self, entry: Dict[str, Any]) -> str:
        """Build a stable signature for an executed action history entry."""
        action_type = entry.get("action_type") or ""
        resource_id = entry.get("resource_id") or ""
        instruction = entry.get("instruction") or ""
        try:
            parsed = json.loads(instruction)
            content = self._extract_primary_content(parsed)
        except Exception:
            # Keep raw instruction when it is truncated or non-JSON
            content = instruction
        return self._normalize_signature(f"{action_type}|{resource_id}|{content}")

    def _is_open_ended_objective(self, objective: str) -> bool:
        """
        Detect objectives that intentionally require ongoing/repeated actions.
        In these cases, do not auto-finalize on repetition.
        """
        text = (objective or "").lower()
        keywords = [
            "monitor",
            "watch continuously",
            "keep checking",
            "real-time",
            "stream",
            "poll",
            "every second",
            "every minute",
            "continuously",
        ]
        return any(k in text for k in keywords)

    def _build_stagnation_response(
        self, state: Dict[str, Any], output_text: str
    ) -> str:
        """
        Build a user-facing final response when stagnation is detected.
        Tries to extract a clean structured answer before falling back to
        the raw output text.
        """
        action_history = state.get("action_history", [])

        # Walk history in reverse to find the most meaningful answer
        for entry in reversed(action_history):
            if not entry.get("success"):
                continue
            result = entry.get("result") or {}
            if isinstance(result, dict):
                answer = (
                    result.get("data", {}).get("answer")
                    or result.get("answer")
                    or result.get("response")
                    or result.get("output")
                )
                if answer and isinstance(answer, str) and len(answer.strip()) > 10:
                    return answer.strip()
            summary = entry.get("result_summary", "")
            if summary and not summary.startswith("{") and len(summary) > 20:
                return summary[:600]

        # Fall back to cleaning the raw output_text (no truncation at 400 chars)
        cleaned = " ".join((output_text or "").split())
        if not cleaned:
            return "Task complete."
        return cleaned[:800] if len(cleaned) > 800 else cleaned

    def _apply_stagnation_guard(
        self, state: Dict[str, Any], decision: BrainDecision
    ) -> BrainDecision:
        """
        Generic loop breaker for repeated successful actions with no observable progress.
        This is intentionally action-agnostic (not tied to a specific use case).
        """
        if decision.action_type in {
            "finish",
            "skip",
            "plan",
            "replan",
            "parallel",
        }:
            return decision

        if self._is_open_ended_objective(state.get("original_prompt", "")):
            return decision

        previous_result = state.get("execution_result") or {}
        if not previous_result.get("success"):
            return decision

        # In plan mode, be slightly more lenient but still detect stagnation.
        # Without a plan: trigger after 3 identical actions. With a plan: 4.
        repeat_threshold = 4 if state.get("execution_plan") else 3

        action_history = list(state.get("action_history", []))
        if len(action_history) < repeat_threshold:
            return decision

        recent = action_history[-repeat_threshold:]
        if not all(entry.get("success") for entry in recent):
            return decision

        recent_signatures = [self._build_history_signature(entry) for entry in recent]
        current_signature = self._build_decision_signature(decision)

        if not current_signature or any(not sig for sig in recent_signatures):
            return decision
        if len(set(recent_signatures + [current_signature])) != 1:
            return decision

        output_text = self._extract_text_output(previous_result.get("output"))
        if not output_text:
            return decision
        lowered = output_text.lower()
        if "[system warning]" in lowered and "no output" in lowered:
            return decision

        logger.warning(
            "Stagnation guard triggered: repeated identical successful action detected; forcing finish."
        )
        return BrainDecision(
            action_type="finish",
            user_response=self._build_stagnation_response(state, output_text),
            reasoning="Repeated identical successful actions detected without new progress; finalized using the latest successful result.",
        )

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

    def _build_created_files_view(self, created_files: List[Dict], workspace_path: str) -> str:
        """Build a view of files created by the orchestrator in this conversation."""
        if not created_files:
            return f"No files created yet in this conversation.\nWorkspace: {workspace_path}"
        
        lines = [f"Workspace: {workspace_path}"]
        for f in created_files:
            name = f.get("file_name", "Unknown")
            ftype = f.get("file_type", "Unknown")
            created_by = f.get("created_by", "Unknown")
            size = f.get("size_bytes", 0)
            size_kb = size / 1024 if size else 0
            
            lines.append(f"- {name} ({ftype}, {size_kb:.1f} KB) [Created by: {created_by}]")
        
        return "\n".join(lines)
    
    def _build_shared_workspace_view(self, shared_files: List[Dict], workspace_path: str) -> str:
        """Build a view of files in the shared workspace."""
        if not shared_files:
            return f"No shared files yet.\nShared workspace: {workspace_path}"
        
        lines = [f"Shared workspace: {workspace_path}"]
        for f in shared_files:
            name = f.get("file_name", "Unknown")
            ftype = f.get("file_type", "Unknown")
            description = f.get("description", "")
            size = f.get("size_bytes", 0)
            size_kb = size / 1024 if size else 0
            
            lines.append(f"- {name} ({ftype}, {size_kb:.1f} KB)")
            if description:
                lines.append(f"  Note: {description}")
        
        return "\n".join(lines)
    
    def _build_agent_workspaces_view(self) -> str:
        """Build a view of files in agent workspaces."""
        try:
            from .workspace_manager import STORAGE_BASE
            if not STORAGE_BASE.exists():
                return "No agent workspaces found."
            
            lines = []
            for agent_dir in STORAGE_BASE.iterdir():
                if agent_dir.is_dir() and agent_dir.name not in ['orchestrator', 'content', 'system', 'vector_store']:
                    files = [f.name for f in agent_dir.iterdir() if f.is_file()][:5]  # Limit to 5
                    if files:
                        lines.append(f"- {agent_dir.name}:")
                        for fname in files:
                            lines.append(f"    - {fname}")
            
            return "\n".join(lines) if lines else "No files in agent workspaces."
        except Exception as e:
            return f"Could not scan agent workspaces: {e}"

    def _sync_execution_plan_to_todo_list(
        self, execution_plan: List[Dict[str, Any]], current_phase_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Convert execution_plan phases into todo_list items for frontend visualization."""
        todo_list = []
        
        for phase in execution_plan:
            phase_id = phase.get("phase_id", "?")
            name = phase.get("name", "Unnamed Phase")
            goal = phase.get("goal", "")
            status_str = phase.get("status", "pending")
            
            # Map phase status to TaskStatus
            if status_str == "completed":
                status = TaskStatus.COMPLETED
            elif phase_id == current_phase_id:
                status = TaskStatus.IN_PROGRESS
            else:
                status = TaskStatus.PENDING
            
            # Use the short phase name as the task title -- goal is full question verbatim
            # and is too long for the task card. Name is already 3-6 words per planner rules.
            description = name
            
            task = TaskItem(
                task_id=f"phase_{phase_id}",
                description=description,
                status=status,
                priority=int(phase_id) if phase_id.isdigit() else 5,
            )
            todo_list.append(task.model_dump())
        
        return todo_list

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

        Reads both sources:
        - last_agent_result (richer: task_summary, answer, tables from agents)
        - execution_result  (Python/tool raw output)
        """
        updated_insights = dict(current_insights)
        iteration = state.get("iteration_count", 0)

        # 1. Agent result -- highest signal, prioritise task_summary / answer fields
        last_agent_result = state.get("last_agent_result")
        if last_agent_result and last_agent_result.get("success"):
            result = last_agent_result.get("result", "")
            agent_name = last_agent_result.get("agent", "agent")
            text = ""
            if isinstance(result, str) and len(result) > 20:
                text = result[:200]
            elif isinstance(result, dict):
                for key in ["task_summary", "answer", "result", "output", "response"]:
                    val = result.get(key)
                    if val and isinstance(val, str) and len(val) > 20:
                        text = val[:200]
                        break
            if text:
                updated_insights[f"agent_{iteration}_{agent_name}"] = text

        # 2. Python / tool execution result
        execution_result = state.get("execution_result", {})
        if execution_result and execution_result.get("success"):
            output = execution_result.get("output")
            if output:
                insight_key = f"step_{iteration}"
                if isinstance(output, dict):
                    for key in ["result", "data", "message", "response", "summary"]:
                        if key in output and output[key]:
                            val = str(output[key])
                            if len(val) > 20:
                                updated_insights[insight_key] = val[:200]
                                break
                elif isinstance(output, str) and len(output) > 20:
                    updated_insights[insight_key] = output[:200]

        return updated_insights

    async def _decompose_into_tasks(
        self, state: Dict[str, Any], config: Optional[RunnableConfig]
    ) -> Dict[str, Any]:
        """
        Mandatory planning step -- runs when only the __initial__ sentinel is present.

        Calls the LLM with a short, focused prompt to decompose the user's request
        into 2-6 concrete, named tasks. The result is applied exactly as if the
        brain had returned action_type='plan', populating the todo_list with real
        phase entries so the UI shows a Manus-style task list from the start.

        Falls back to a single generic task if the LLM fails.
        """
        prompt = state.get("original_prompt", "")
        uploaded = state.get("uploaded_files", [])
        files_note = ""
        if uploaded:
            names = ", ".join(f.get("file_name", "?") for f in uploaded[:5])
            files_note = f"\nUploaded files available: {names}"

        # Get available agents for context
        try:
            from backend.services.agent_registry_service import agent_registry
            agents = agent_registry.list_active_agents()
            agent_names = ", ".join(a["id"] for a in agents) if agents else "python, terminal"
        except Exception:
            agent_names = "python, terminal"

        planning_prompt = f"""You are a task planner. Break the user's request into 2 to 6 concrete, specific steps.

USER REQUEST:
{prompt}{files_note}

AVAILABLE AGENTS/TOOLS: {agent_names}

Output a JSON array of steps. Each step must have:
- "phase_id": "1", "2", ... (string numbers)
- "name": short label (3-6 words, e.g. "Fetch supplier data")
- "goal": one sentence describing exactly what this step does
- "depends_on": [] for first step, ["1"] for second, etc.

Rules:
- 2 steps minimum, 6 steps maximum
- Every step must be a concrete, named action -- NOT vague ("Process data", "Gather info")
- Name the actual operation: "Calculate top 5 suppliers by spend", "Generate bar chart of results"
- The last step should always be "Compile and present results" or similar

CRITICAL RULE FOR SINGLE-AGENT TASKS (spreadsheet, document, data analysis):
- If the task requires a specialized agent (spreadsheet analysis, document reading, etc.), create EXACTLY 2 phases:
  1. One phase where the agent does ALL the analysis -- set the goal to the COMPLETE user question verbatim
  2. One phase: "Compile and present results"
- Do NOT split a single-agent task into sub-steps like "Load file", "Extract columns", "Compute X", "Compute Y".
  The agent answers everything in ONE call when given the full question. Splitting forces multiple agent calls
  and breaks the analysis.

Respond with ONLY the JSON array, no explanation.

Example for "summarise a PDF and email it":
[
  {{"phase_id": "1", "name": "Read and summarise PDF", "goal": "Extract key points from the uploaded document", "depends_on": []}},
  {{"phase_id": "2", "name": "Draft email body", "goal": "Write a concise email containing the summary", "depends_on": ["1"]}},
  {{"phase_id": "3", "name": "Send email", "goal": "Deliver the email to the specified recipient", "depends_on": ["2"]}}
]"""

        phases = None
        try:
            result = await inference_service.generate_structured(
                messages=[HumanMessage(content=planning_prompt)],
                schema=_PlanningOutput,
                priority=InferencePriority.HIGH,
                max_tokens=800,
                telemetry_metadata={
                    "agent_name": "Brain",
                    "operation_type": "task_planning",
                },
            )
            if result.phases:
                phases = [p.model_dump() for p in result.phases]
        except Exception as e:
            logger.warning(f"Planning LLM call failed, using fallback: {e}")

        if not phases:
            # Fallback: two-step plan so the UI always shows something meaningful
            phases = [
                {"phase_id": "1", "name": "Complete request", "goal": prompt[:200], "depends_on": []},
                {"phase_id": "2", "name": "Compile and present results", "goal": "Present the final answer to the user", "depends_on": ["1"]},
            ]

        # Apply as a plan decision -- reuse existing plan→todo_list machinery
        fake_decision = BrainDecision(
            action_type="plan",
            execution_plan=phases,
            reasoning="Decomposed request into concrete steps",
        )
        updates = self._apply_decision_to_state(state, fake_decision)
        updates["iteration_count"] = state.get("iteration_count", 0)
        logger.info(f"Decomposed into {len(phases)} tasks: {[p.get('name') for p in phases]}")
        return updates

    def _initialize_initial_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize the todo list with a sentinel placeholder task.

        The sentinel task_id '__initial__' is detected by the brain prompt on the
        very next think() call, which forces action_type='plan' to decompose the
        request into concrete steps before any execution begins.
        """
        initial_task = TaskItem(
            task_id="__initial__",
            description="Analysing request…",
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
            "current_task_id": initial_task.task_id,
            "decision": BrainDecision(
                action_type="skip",
                reasoning="Initializing system state",
            ).model_dump(),
        }


    # -----------------------------------------------------------------------
    # Connection-gate: check Composio connections before agent delegation
    # -----------------------------------------------------------------------

    # Maps agent resource_id patterns → (app_slug, display_name)
    _AGENT_APP_MAP: Dict[str, tuple] = {
        "gmail": ("gmail", "Gmail"),
        "gmail_agent": ("gmail", "Gmail"),
        "mail": ("gmail", "Gmail"),
        "mail_agent": ("gmail", "Gmail"),
        "zoho": ("zohobooks", "Zoho Books"),
        "zoho_books": ("zohobooks", "Zoho Books"),
        "zoho_books_agent": ("zohobooks", "Zoho Books"),
    }

    async def _check_connection_for_decision(
        self, state: Dict[str, Any], decision: BrainDecision
    ) -> BrainDecision:
        """
        Connection-gate: if the Brain has decided to route to an agent that needs
        a Composio connection, verify that the user has an active connection first.

        If the connection is MISSING:
          - Rewrite the decision to use `integrations_agent` with
            `requires_approval=True` so the frontend shows an OAuth link.

        If the connection is PRESENT (or not required):
          - Return the decision unchanged.
        """
        if decision.action_type != "agent":
            return decision
        
        # Skip connection check if we're already executing integrations_agent
        # This happens after user approves the OAuth dialog - don't redirect infinitely
        if decision.resource_id and "integrations" in (decision.resource_id or "").lower():
            logger.info(f"Skipping connection-gate for integrations_agent to avoid infinite redirect")
            return decision

        resource_id = (decision.resource_id or "").lower().strip()
        # Resolve via alias map
        app_key = resource_id
        for key in self._AGENT_APP_MAP:
            if key in resource_id:
                app_key = key
                break

        app_info = self._AGENT_APP_MAP.get(app_key)
        if not app_info:
            # This agent doesn't need a specific Composio connection
            return decision

        app_slug, app_display_name = app_info
        user_id = state.get("user_id") or state.get("owner_id") or "default"

        try:
            from services.integrations.composio_auth import get_auth_manager

            auth_mgr = get_auth_manager()
            
            # Sync from Composio first — after OAuth, the DB may still show INITIATED
            # while Composio already has the connection as ACTIVE. This sync updates the DB.
            auth_mgr.check_connection_status(user_id, app_slug)
            
            connection = auth_mgr.get_connection_for_agent(user_id, app_slug)

            if connection:
                # Connection found – proceed normally, inject user_id into payload
                payload = decision.payload or {}
                payload.setdefault("user_id", user_id)
                return BrainDecision(
                    **{
                        **decision.model_dump(),
                        "payload": payload,
                    }
                )

            # Connection missing – redirect to integrations_agent with OAuth prompt
            logger.info(
                f"🔌 Brain connection-gate: {app_slug} not connected for "
                f"user={user_id} -- redirecting to integrations_agent"
            )

            original_task = (
                (decision.payload or {}).get("prompt")
                or (decision.payload or {}).get("instruction")
                or state.get("original_prompt", "")
            )

            return BrainDecision(
                action_type="agent",
                resource_id="Integrations Agent",
                requires_approval=True,
                approval_reason=(
                    f"Need to connect {app_display_name} to complete this task. "
                    f"Click 'Approve' to authorise and continue."
                ),
                payload={
                    "user_id": user_id,
                    "prompt": original_task,
                    "app_name": app_slug,
                    "requesting_service": app_slug,
                    "post_auth_task": original_task,
                },
                reasoning=(
                    f"User {user_id} has no active {app_display_name} connection. "
                    "Routing to Integrations Agent to initiate in-chat OAuth."
                ),
            )

        except Exception as e:
            logger.warning(
                f"Brain connection-gate error (non-fatal, proceeding): {e}"
            )
            return decision

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
            "pending_user_input": False,
            "user_response": None
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
            # Sync execution_plan to todo_list for frontend visualization
            updates["todo_list"] = self._sync_execution_plan_to_todo_list(
                decision.execution_plan, 
                updates.get("current_phase_id")
            )
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

            # Sync replanned execution_plan to todo_list
            updates["todo_list"] = self._sync_execution_plan_to_todo_list(
                new_plan, 
                updates.get("current_phase_id")
            )

            logger.info(
                f"🔄 Re-planned: kept {len(completed_phases)} completed, added {len(validated_plan)} new phases"
            )

        # Handle PARALLEL action - just pass through, Hands will execute
        # No special state handling needed here

        # Handle direct execution actions
        if decision.action_type not in ("finish", "skip", "plan", "replan", "parallel"):
            # If current_task_id is stale (sentinel or doesn't exist in current todo_list),
            # reset it so the next block picks the first pending task.
            existing_ids = {t.get("task_id") for t in todo_list}
            if current_task_id not in existing_ids:
                current_task_id = None

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

            # Resync todo_list after phase progression
            updates["todo_list"] = self._sync_execution_plan_to_todo_list(
                updated_plan, 
                next_phase_id
            )

            logger.info(
                f"✅ Phase '{current_phase_id}' verified complete by LLM → Next: {next_phase_id or 'ALL DONE'}"
            )

        # Handle finish
        if decision.action_type == "finish":
            updates["final_response"] = decision.user_response or "Task complete."

            # === CANVAS AUTO-DETECT for finish responses ===
            # When Brain finishes directly (bypassing Hands), detect structured content
            # and register canvas in the Canvas Registry
            user_resp = decision.user_response or ""
            if len(user_resp) > 300:
                try:
                    import re
                    from services.canvas_service import CanvasService
                    from backend.services.canvas_registry import get_canvas_registry

                    thread_id = (
                        state.get("thread_id")
                        or state.get("configurable", {}).get("thread_id", "default")
                    )

                    canvas_obj = None

                    # 1. Code block detection (5+ lines)
                    code_match = re.search(r'```(\w+)?\n(.*?)```', user_resp, re.DOTALL)
                    if code_match:
                        lang = code_match.group(1) or 'python'
                        code = code_match.group(2).strip()
                        if code and code.count('\n') >= 4:
                            canvas_obj = CanvasService.build_from_template(
                                "code_viewer",
                                {"code": code, "language": lang},
                                title=f"Code ({lang})",
                            )

                    # 2. Long markdown with structure → document_viewer
                    if not canvas_obj:
                        has_headers = bool(re.search(r'^##?\s+', user_resp, re.MULTILINE))
                        has_tables = '|---' in user_resp or '| ---' in user_resp
                        if has_headers or has_tables:
                            title_match = re.search(r'^#\s+(.+)', user_resp, re.MULTILINE)
                            title = title_match.group(1).strip() if title_match else "Document"
                            canvas_obj = CanvasService.build_from_template(
                                "document_viewer",
                                {"content": user_resp, "title": title, "status": "created"},
                                title=title,
                            )

                    # Register canvas in registry
                    if canvas_obj:
                        canvas_dict = canvas_obj.model_dump()
                        registry = get_canvas_registry(thread_id)
                        import time as _time
                        c_type = canvas_dict.get("canvas_type", "document")
                        canvas_id = f"brain_finish_{c_type}_{int(_time.time())}"
                        registry.register_sync(
                            canvas_id=canvas_id,
                            canvas_type=c_type,
                            source_agent="brain",
                            canvas_data=canvas_dict.get("canvas_data"),
                            canvas_content=canvas_dict.get("canvas_content"),
                            canvas_title=canvas_dict.get("canvas_title"),
                        )
                        compat = registry.get_backward_compat_fields()
                        updates.update(compat)
                        updates["canvas_registry"] = registry.get_registry_state().model_dump()
                        updates["active_canvas_id"] = registry.get_active_id()
                        logger.info(f"🎨 Brain: Auto-canvas registered for finish response (type={c_type})")

                except Exception as e:
                    logger.debug(f"Canvas auto-detect in finish skipped: {e}")

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
                    task["status"] = "completed"
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

    def _build_failure_guidance(self, state: Dict[str, Any]) -> str:
        """
        Build dynamic guidance for the LLM based on current failure patterns.
        Instead of hardcoded limits, this lets the LLM reason about failures.
        """
        failure_count = state.get("failure_count", 0)
        if failure_count == 0:
            return ""
        
        action_history = state.get("action_history", [])
        
        # Analyze recent failures to detect patterns
        recent_failures = []
        for entry in reversed(action_history):
            if not entry.get("success"):
                recent_failures.append(entry)
            else:
                break  # Stop at first success (we only want the consecutive tail)
        
        # Build specific guidance based on failure patterns
        guidance_parts = []
        
        if failure_count >= 2:
            guidance_parts.append(
                f"⚠️ You have {failure_count} consecutive failures. You MUST change your approach -- do NOT repeat the same action."
            )
        
        # Detect common failure patterns
        error_messages = [f.get("result_summary", "") for f in recent_failures]
        error_types = set()
        for err in error_messages:
            err_lower = err.lower()
            if "403" in err or "forbidden" in err_lower:
                error_types.add("http_blocked")
            if "timeout" in err_lower or "timed out" in err_lower:
                error_types.add("timeout")
            if "connection" in err_lower:
                error_types.add("connection")
            if "agent" in err_lower or any(f.get("action_type") == "agent" for f in recent_failures):
                error_types.add("agent_failure")
        
        if "http_blocked" in error_types:
            guidance_parts.append(
                "🔄 HTTP requests are being blocked (403). Use `requests.get(url)` which has proper headers, "
                "or try `read_html(url)` for HTML tables. Do NOT use urllib directly."
            )
        if "agent_failure" in error_types:
            failed_agents = set(f.get("resource_id", "") for f in recent_failures if f.get("action_type") == "agent")
            guidance_parts.append(
                f"🔄 Agent(s) {failed_agents} failed. Consider using Python or tools as a fallback instead of retrying the same agent."
            )
        if "timeout" in error_types:
            guidance_parts.append(
                "🔄 Timeouts detected. Try simpler queries or break the task into smaller parts."
            )
        
        if failure_count >= 3:
            guidance_parts.append(
                "💡 Consider: Can you complete the task with the data you already have? "
                "If so, use action_type='finish' to provide the best possible answer. "
                "If not, try a fundamentally different approach."
            )
        
        return "\n".join(guidance_parts) if guidance_parts else ""

    def _enter_fallback_mode(
        self,
        state: Dict[str, Any],
        memory: Dict[str, Any],
        insights: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """Fallback mode: provide best answer from available context."""
        # Build a useful summary from action history instead of raw JSON
        action_history = state.get("action_history", [])
        successes = []
        failures = []
        for entry in action_history:
            action_desc = f"{entry.get('action_type', '?')}:{entry.get('resource_id', '?')}"
            if entry.get("success"):
                summary = entry.get("result_summary", "")[:150]
                successes.append(f"- ✅ {action_desc}: {summary}")
            else:
                error = entry.get("result_summary", "")[:100]
                failures.append(f"- ❌ {action_desc}: {error}")
        
        parts = ["I encountered multiple issues while working on your request. Here's what happened:\n"]
        if successes:
            parts.append("**Completed steps:**")
            parts.extend(successes[:10])
        if failures:
            parts.append("\n**Failed steps:**")
            parts.extend(failures[:5])
        
        # Include any insights gathered
        if insights:
            parts.append("\n**What I learned:**")
            for key, val in list(insights.items())[:5]:
                parts.append(f"- {str(val)[:150]}")
        
        parts.append("\nPlease try again or rephrase your request. If a specific step keeps failing, I can try an alternative approach.")
        
        user_response = "\n".join(parts)
        
        return {
            "decision": BrainDecision(
                action_type="finish",
                user_response=user_response,
            ).model_dump(),
            "final_response": user_response,
            "current_task_id": None,
        }

    def _force_finish_with_error(
        self, state: Dict[str, Any], error: str
    ) -> Dict[str, Any]:
        """
        Force the graph to terminate by emitting action_type='finish'.

        Previously this only set final_response without updating `decision`,
        so omni_route_condition never saw action_type='finish' and the graph
        kept looping indefinitely.  This fix makes the routing condition see
        the finish signal and route to END.

        When the graph has collected successful action results we attempt to
        surface the most useful one as the user-facing response rather than
        dumping raw internal summaries.
        """
        action_history = state.get("action_history", [])

        # Try to find a meaningful answer from the last successful action
        user_response = None
        for entry in reversed(action_history):
            if not entry.get("success"):
                continue
            # Prefer structured data fields over raw summary strings
            result = entry.get("result") or {}
            if isinstance(result, dict):
                # agent results often have data.answer or answer
                answer = (
                    result.get("data", {}).get("answer")
                    or result.get("answer")
                    or result.get("response")
                    or result.get("output")
                )
                if answer and isinstance(answer, str) and len(answer.strip()) > 10:
                    user_response = answer.strip()
                    break
            # Fall back to result_summary if it looks like prose (not a dict repr)
            summary = entry.get("result_summary", "")
            if summary and not summary.startswith("{") and len(summary) > 20:
                user_response = summary[:500]
                break

        if not user_response:
            original_prompt = state.get("original_prompt", "")
            user_response = (
                "I was unable to complete the task within the allowed steps. "
                "Please try rephrasing your request or breaking it into smaller parts."
            )
            if original_prompt:
                user_response += f'\n\nOriginal request: "{original_prompt[:200]}"'

        finish_decision = BrainDecision(
            action_type="finish",
            user_response=user_response,
            reasoning=error,
        )
        return {
            "final_response": user_response,
            "decision": finish_decision.model_dump(),
            "current_task_id": None,
        }
