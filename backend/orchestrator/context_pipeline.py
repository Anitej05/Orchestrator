"""
Context Pipeline — Modular Context Assembly for Brain Decisions

Replaces the monolithic ~500-line prompt in Brain._make_decision() with
a pipeline that assembles only relevant context pieces, each with a token budget.

Also provides isolated context assembly for agent dispatch (context isolation).

Design principles:
- Each section has a configurable token budget
- CMS integration for content references (not raw data)
- ArtifactStore integration for past experience
- SkillRegistry integration for compact agent summaries
- Progressive disclosure: agents get only what they need
"""

import logging
import json
from typing import Dict, Any, Optional, List

logger = logging.getLogger("ContextPipeline")


# ──────────────────────────────────────────────────────────────────────────
# TOKEN ESTIMATION
# ──────────────────────────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return len(text) // 4


def _truncate_to_budget(text: str, max_tokens: int) -> str:
    """Truncate text to fit within a token budget, adding truncation marker."""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 40] + "\n... [truncated — see full via services]"


# ──────────────────────────────────────────────────────────────────────────
# SECTION BUILDERS — Each builds one section of the Brain prompt
# ──────────────────────────────────────────────────────────────────────────

def _build_objective_section(state: Dict[str, Any], budget: int) -> str:
    """The user's original prompt / objective."""
    prompt = state.get("original_prompt", "No objective")
    return _truncate_to_budget(f"## OBJECTIVE\n{prompt}", budget)


def _build_skills_section(state: Dict[str, Any], budget: int) -> str:
    """Compact skill summary using semantic matching."""
    try:
        from backend.services.skill_registry import skill_registry
        skill_registry.initialize()
        
        objective = state.get("original_prompt", "")
        if objective:
            # Semantic pre-filtering: find top 5 most relevant agents for this task
            matches = skill_registry.match_skills(objective, top_k=5)
            if not matches:
                return "## AVAILABLE AGENTS\nNo relevant agents found."
                
            lines = []
            for m in matches:
                config = skill_registry.get_skill_config(m.skill_id)
                if config:
                    triggers_str = ", ".join(config.triggers[:5]) if config.triggers else "general"
                    lines.append(
                        f"- **{config.name}** (id: `{m.skill_id}`): "
                        f"{config.description[:100]}... "
                        f"[triggers: {triggers_str}]"
                    )
            summary = "\n".join(lines)
        else:
            # Fallback if no objective: show all (legacy behavior)
            summary = skill_registry.get_skill_summary()
            
        return _truncate_to_budget(
            f"## AVAILABLE AGENTS\n{summary}", budget
        )
    except Exception as e:
        logger.debug(f"SkillRegistry unavailable: {e}")
        return "## AVAILABLE AGENTS\nNo agents available."


def _build_tools_section(budget: int) -> str:
    """Compact tool summary from ToolRegistry."""
    try:
        from backend.services.tool_registry_service import tool_registry
        summary = tool_registry.get_tool_prompt_summary()
        return _truncate_to_budget(
            f"## AVAILABLE TOOLS\n{summary}", budget
        )
    except Exception as e:
        logger.debug(f"ToolRegistry unavailable: {e}")
        return "## AVAILABLE TOOLS\nNo tools available."


def _build_memory_section(state: Dict[str, Any], budget: int) -> str:
    """Memory + insights from state."""
    memory = state.get("memory", {})
    insights = state.get("insights", {})

    parts = ["## MEMORY & INSIGHTS"]

    if memory:
        mem_str = json.dumps(memory, indent=2, default=str)
        parts.append(f"### Working Memory\n{mem_str}")

    if insights:
        parts.append("### Key Insights")
        for key, value in insights.items():
            parts.append(f"- **{key}**: {value}")

    if len(parts) == 1:
        parts.append("No memory or insights yet.")

    return _truncate_to_budget("\n".join(parts), budget)


def _build_action_history_section(
    action_history: List[Dict], budget: int
) -> str:
    """Compact action history with dynamic budget."""
    if not action_history:
        return "## ACTION HISTORY\nNo actions taken yet."

    parts = ["## ACTION HISTORY"]
    
    # Show most recent actions first (most relevant)
    for i, entry in enumerate(reversed(action_history)):
        action_type = entry.get("action_type", "unknown")
        resource = entry.get("resource_id", "")
        success = "✅" if entry.get("success", False) else "❌"
        summary = entry.get("result_summary", entry.get("output_text", ""))

        # Trim individual summaries
        if isinstance(summary, str) and len(summary) > 300:
            summary = summary[:300] + "..."

        parts.append(
            f"\n### [{len(action_history) - i}] {success} {action_type}"
            f"{'(' + resource + ')' if resource else ''}"
        )
        if summary:
            parts.append(f"```\n{summary}\n```")

        # Check budget as we go
        current = "\n".join(parts)
        if _estimate_tokens(current) > budget * 0.9:
            parts.append(f"\n... ({i + 1}/{len(action_history)} actions shown)")
            break

    return _truncate_to_budget("\n".join(parts), budget)


def _build_files_section(state: Dict[str, Any], budget: int) -> str:
    """Uploaded and created files — compact listing."""
    uploaded = state.get("uploaded_files", [])
    created = state.get("created_files", [])

    parts = ["## FILES"]

    if uploaded:
        parts.append("### Uploaded")
        for f in uploaded:
            name = f.get("name", f.get("filename", "unknown"))
            path = f.get("path", f.get("file_path", ""))
            parts.append(f"- `{name}` → `{path}`")

    if created:
        parts.append("### Created This Session")
        for f in created[:10]:  # Cap at 10
            name = f.get("name", f.get("filename", ""))
            parts.append(f"- `{name}`")

    if not uploaded and not created:
        parts.append("No files available.")

    return _truncate_to_budget("\n".join(parts), budget)


def _build_conversation_section(
    messages: List[Any], budget: int
) -> str:
    """Recent conversation messages — compact."""
    if not messages:
        return "## CONVERSATION\nNo conversation history."

    parts = ["## RECENT CONVERSATION"]

    # Show last 6 messages max
    for msg in messages[-6:]:
        msg_type = msg.__class__.__name__ if hasattr(msg, '__class__') else "Message"
        content = msg.content if hasattr(msg, 'content') else str(msg)
        
        # Compact representation
        role = "🧑 User" if "Human" in msg_type else "🤖 Assistant"
        if len(content) > 200:
            content = content[:200] + "..."
        parts.append(f"**{role}**: {content}")

    return _truncate_to_budget("\n".join(parts), budget)


def _build_artifacts_section(state: Dict[str, Any], budget: int) -> str:
    """Relevant past experience from ArtifactStore."""
    try:
        from backend.orchestrator.artifact_store import get_artifact_store
        user_id = state.get("user_id", "default")
        store = get_artifact_store(user_id)
        artifacts_str = store.retrieve_relevant(
            query=state.get("original_prompt", ""),
            top_k=3,
            max_tokens=budget * 3,  # char budget ~ 3x token budget
        )
        if artifacts_str:
            return _truncate_to_budget(
                f"## RELEVANT EXPERIENCE\n{artifacts_str}", budget
            )
    except Exception as e:
        logger.debug(f"ArtifactStore unavailable: {e}")

    return "## RELEVANT EXPERIENCE\nNo relevant past experience."


def _build_todo_section(state: Dict[str, Any], budget: int) -> str:
    """Current task list — compact."""
    todo_list = state.get("todo_list", [])
    if not todo_list:
        return "## TO-DO LIST\nNo tasks."

    parts = ["## TO-DO LIST"]
    for task in todo_list:
        status = task.get("status", "pending")
        icon = {"completed": "✅", "in_progress": "🔄", "failed": "❌"}.get(
            status, "⬜"
        )
        parts.append(f"{icon} {task.get('description', task.get('task_id', '?'))}")

    return _truncate_to_budget("\n".join(parts), budget)


def _build_plan_section(state: Dict[str, Any], budget: int) -> str:
    """Execution plan if exists."""
    plan = state.get("execution_plan")
    if not plan:
        return ""

    parts = ["## EXECUTION PLAN"]
    phases = plan if isinstance(plan, list) else plan.get("phases", [])
    for phase in phases:
        status = phase.get("status", "pending")
        icon = {"completed": "✅", "in_progress": "🔄"}.get(status, "⬜")
        parts.append(f"{icon} Phase {phase.get('phase_id', '?')}: {phase.get('name', '')}")
        if phase.get("goal"):
            parts.append(f"   Goal: {phase['goal']}")

    return _truncate_to_budget("\n".join(parts), budget)


# ──────────────────────────────────────────────────────────────────────────
# DECISION RULES — Concise, de-duplicated
# ──────────────────────────────────────────────────────────────────────────

DECISION_RULES = """## DECISION RULES

1. **Match step → resource:** Use agent skills above to pick the right agent. Use tools for quick lookups. Use Python for computation.
2. **Agent-first for files:** If uploaded files exist AND a matching agent is available, call the agent FIRST with the COMPLETE question — not a schema probe.
3. **No redundancy:** Never repeat the same action. Previous results are in ACTION HISTORY above.
4. **Trust agent results:** If an agent returned success, DO NOT re-verify with Python. Use the result directly.
5. **Finish when done:** Once the objective is fully met, call `action_type='finish'` with a formatted response.
6. **Plan before execute:** If TO-DO has only `__initial__`, respond with `action_type='plan'` first.
7. **Use code in action_type='python':** Code in `user_response` is NOT executed. Only `action_type='python'` runs code.
8. **Files need action_type='python' or 'terminal':** Files are NOT created by `finish`.
9. **Handle errors:** If something fails, try a different approach — don't repeat the failing action.
"""


# ══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════

class ContextPipeline:
    """
    Assembles optimized context for Brain decisions and agent dispatch.
    
    Each section has a configurable token budget. The total prompt is
    significantly smaller than the current monolithic ~500-line version.
    """

    # Default token budgets per section
    DEFAULT_BUDGETS = {
        "objective": 200,
        "skills": 400,
        "tools": 300,
        "memory": 400,
        "action_history": 1500,
        "files": 200,
        "conversation": 400,
        "artifacts": 400,
        "todo": 200,
        "plan": 200,
        "rules": 400,       # decision rules (fixed text)
    }

    def __init__(self, budgets: Optional[Dict[str, int]] = None):
        self.budgets = {**self.DEFAULT_BUDGETS, **(budgets or {})}

    def assemble(
        self,
        state: Dict[str, Any],
        config: Optional[Dict] = None,
    ) -> str:
        """
        Build the full Brain prompt from modular sections.
        
        Returns a focused prompt (~200 lines vs current ~500 lines).
        Each section is independently budgeted.
        """
        sections = []

        # Persona (fixed, small)
        sections.append(
            "You are the Brain of an intelligent orchestrator. "
            "Achieve the objective by selecting the best resource for each step."
        )

        # Core sections
        sections.append(_build_objective_section(state, self.budgets["objective"]))
        sections.append(_build_skills_section(state, self.budgets["skills"]))
        sections.append(_build_tools_section(self.budgets["tools"]))
        sections.append(_build_files_section(state, self.budgets["files"]))
        sections.append(_build_memory_section(state, self.budgets["memory"]))

        # Plan and tasks
        plan = _build_plan_section(state, self.budgets["plan"])
        if plan:
            sections.append(plan)
        sections.append(_build_todo_section(state, self.budgets["todo"]))

        # History (largest budget — most important for avoiding loops)
        sections.append(
            _build_action_history_section(
                state.get("action_history", []),
                self.budgets["action_history"],
            )
        )

        # Conversation
        sections.append(
            _build_conversation_section(
                state.get("messages", []),
                self.budgets["conversation"],
            )
        )

        # Past experience
        sections.append(_build_artifacts_section(state, self.budgets["artifacts"]))

        # Iteration awareness
        iteration = state.get("iteration_count", 0)
        max_iter = 50  # from Brain._MAX_ITERATIONS
        sections.append(
            f"## STATUS\nIteration: {iteration}/{max_iter} | "
            f"Failures: {state.get('failure_count', 0)}"
        )

        # Latest agent result (prominent, for fast finish)
        last_result = state.get("last_agent_result")
        if last_result and last_result.get("success"):
            sections.append(
                f"## LATEST RESULT -- CONSIDER FINISHING\n"
                f"Agent: **{last_result.get('agent', 'unknown')}**\n"
                f"```\n{str(last_result.get('result', ''))[:500]}\n```\n"
                f"If the objective is now met, call action_type='finish' immediately."
            )

        # Decision rules (fixed text, always last)
        sections.append(DECISION_RULES)

        return "\n\n".join(s for s in sections if s)

    def assemble_for_agent(
        self,
        agent_id: str,
        task_prompt: str,
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build isolated context for agent dispatch.
        
        Only includes: task prompt, relevant files, user_id.
        Does NOT include: full state, action history, other agent info.
        This enforces context isolation — the agent works independently.
        """
        # Only pass relevant files to the agent
        relevant_files = []
        for f in state.get("uploaded_files", []):
            relevant_files.append({
                "name": f.get("name", f.get("filename", "")),
                "path": f.get("path", f.get("file_path", "")),
                "type": f.get("type", f.get("content_type", "")),
            })

        payload = {
            "prompt": task_prompt,
            "user_id": state.get("user_id", "default"),
            "thread_id": state.get("thread_id", "default"),
        }

        # Include file references if any
        if relevant_files:
            payload["files"] = relevant_files
            # Also include file_path for backward compatibility
            if len(relevant_files) == 1:
                payload["file_path"] = relevant_files[0].get("path", "")

        # Get agent's skill context (full body) for enrichment
        try:
            from backend.services.skill_registry import skill_registry
            skill_config = skill_registry.get_skill_config(agent_id)
            if skill_config and skill_config.context_strategy == "full":
                # Full context agents get skill instructions too
                payload["skill_context"] = skill_registry.get_skill_context(agent_id)
        except Exception:
            pass

        return payload
