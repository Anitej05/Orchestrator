"""
Unit tests for backend/orchestrator/brain.py

Tests the Brain's deterministic logic without hitting real LLM endpoints.
All LLM calls are mocked via patch.object / AsyncMock.

Coverage:
  1.  BrainDecision model parsing + defaults
  2.  action_type='agent' resource_id round-trip
  3.  action_type='finish' writes final_response
  4.  requires_approval sets pending_approval (early-return, no execution)
  5.  Stagnation guard — triggers finish on repeated identical calls
  6.  Stagnation guard — does NOT fire on diverse actions
  7.  Stagnation guard — skips open-ended / monitoring objectives
  8.  Stagnation guard — does NOT fire when last result was a failure
  9.  Adaptive budget — simple prompt → 12 iterations
  10. Adaptive budget — complex prompt → 25 iterations
  11. Adaptive budget — medium prompt → 18 iterations
  12. Adaptive budget — mixed signals → medium (not simple)
  13. Iteration limit forces finish
  14. force_finish surfaces meaningful answer from history
  15. force_finish provides fallback when no history
  16. _apply_decision_to_state: plan → execution_plan + todo_list sync
  17. _apply_decision_to_state: agent → marks task in_progress
  18. _apply_decision_to_state: finish → all tasks completed
  19. _apply_decision_to_state: phase_complete=True advances phase
  20. _apply_decision_to_state: last phase done → current_phase_id=None
  21. _apply_decision_to_state: memory_updates merged
  22. _apply_decision_to_state: iteration_count incremented
  23. Execution plan validation — well-formed, no errors
  24. Execution plan validation — auto-assigns missing phase_id
  25. Execution plan validation — auto-assigns missing name
  26. Execution plan validation — flags invalid dependency
  27. Execution plan validation — sets pending status
  28. _sync_execution_plan_to_todo_list: pending / in_progress / completed status
  29. _sync_execution_plan_to_todo_list: task_id has phase_ prefix
  30. _sync_execution_plan_to_todo_list: description = name, not goal
  31. _extract_insights_from_last_action: string output
  32. _extract_insights_from_last_action: dict result field
  33. _extract_insights_from_last_action: ignores failures
  34. _extract_insights_from_last_action: ignores short output
  35. _initialize_initial_state: creates __initial__ sentinel
  36. _initialize_initial_state: resets action_history and memory
  37. _initialize_initial_state: decision is skip
  38. think(): empty todo_list → creates initial state
  39. think(): __initial__ sentinel → calls _decompose_into_tasks
  40. think(): iteration_count == max_iterations → forces finish
  41. think(): Python-level MANDATORY FINISH after agent success on auto-phases
  42. Replan: preserves completed phases, drops old pending, adds new phases
  43. Replan: current_phase_id points to first non-completed phase
  44. Build helpers: _build_todo_preview, _build_insights_view, etc.
  45. Artifact store RAG: retrieve_relevant is queried during _make_decision
  46. Safety: LLM refusal for illegal backdating passes through correctly
  47. Safety: LLM refusal for pre-QC CoA dispatch passes through correctly
  48. Safety: clarification on ambiguous prompt passes through correctly
"""

import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent       # backend/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))               # project root

from backend.orchestrator.brain import Brain, BrainDecision
from backend.orchestrator.schemas import TaskItem, TaskStatus


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_task(task_id: str, status: str = "pending") -> Dict:
    return TaskItem(
        task_id=task_id,
        description=f"Task: {task_id}",
        status=TaskStatus(status),
        priority=1,
    ).model_dump()


def _base_state(**overrides) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "original_prompt": "What is 2+2?",
        "todo_list": [_make_task("task_1")],
        "memory": {},
        "insights": {},
        "action_history": [],
        "iteration_count": 1,
        "failure_count": 0,
        "execution_result": None,
        "last_agent_result": None,
        "execution_plan": None,
        "current_phase_id": None,
        "current_task_id": "task_1",
        "uploaded_files": [],
        "messages": [],
        "user_id": "test_user",
    }
    state.update(overrides)
    return state


def _history_entry(
    action_type: str = "python",
    resource_id: str = "",
    instruction: str = '{"code": "print(1)"}',
    success: bool = True,
    result_summary: str = "Output: 1",
    result: Any = None,
    iteration: int = 1,
) -> Dict:
    return {
        "action_type": action_type,
        "resource_id": resource_id,
        "instruction": instruction,
        "success": success,
        "result_summary": result_summary,
        "result": result or {},
        "iteration": iteration,
    }


def _repeated_history(n: int, action_type="python", resource_id="") -> List[Dict]:
    instruction = json.dumps({"code": "print('hello')"})
    return [
        _history_entry(
            action_type=action_type,
            resource_id=resource_id,
            instruction=instruction,
            success=True,
            result_summary="Output: hello",
            iteration=i,
        )
        for i in range(1, n + 1)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 1-4  BrainDecision model
# ─────────────────────────────────────────────────────────────────────────────

class TestBrainDecisionModel:

    def test_brain_decision_parsed_correctly(self):
        """BrainDecision parses from a plain dict with all fields."""
        data = {
            "action_type": "agent",
            "resource_id": "SpreadsheetAgent",
            "payload": {"prompt": "Summarise sales data"},
            "reasoning": "File uploaded, use agent",
            "requires_approval": False,
        }
        d = BrainDecision(**data)
        assert d.action_type == "agent"
        assert d.resource_id == "SpreadsheetAgent"
        assert d.payload == {"prompt": "Summarise sales data"}
        assert d.requires_approval is False

    def test_brain_decision_defaults(self):
        """BrainDecision initialises safe defaults for every optional field."""
        d = BrainDecision(action_type="finish", user_response="Done")
        assert d.requires_approval is False
        assert d.phase_complete is False
        assert d.resource_id is None
        assert d.execution_plan is None
        assert d.parallel_actions is None

    def test_action_type_agent_sets_resource_id(self):
        """Agent decisions carry resource_id; it round-trips through model_dump."""
        d = BrainDecision(
            action_type="agent",
            resource_id="DocumentAgent",
            payload={"prompt": "Summarise the PDF"},
        )
        dumped = d.model_dump()
        assert dumped["action_type"] == "agent"
        assert dumped["resource_id"] == "DocumentAgent"

    def test_action_type_finish_sets_user_response(self):
        """Finish decision → _apply_decision_to_state stores it in final_response."""
        brain = Brain()
        state = _base_state()
        d = BrainDecision(action_type="finish", user_response="The answer is 42.")
        updates = brain._apply_decision_to_state(state, d)
        assert updates["final_response"] == "The answer is 42."
        assert updates["decision"]["action_type"] == "finish"

    def test_requires_approval_set_for_sensitive_actions(self):
        """requires_approval=True → pending_approval in state; no execution yet."""
        brain = Brain()
        state = _base_state()
        d = BrainDecision(
            action_type="agent",
            resource_id="MailAgent",
            payload={"prompt": "Send report to cfo@company.com"},
            requires_approval=True,
            approval_reason="Will send email to cfo@company.com",
        )
        updates = brain._apply_decision_to_state(state, d)
        assert updates.get("pending_approval") is True
        assert updates.get("pending_decision") is not None
        assert "final_response" not in updates  # execution not started


# ─────────────────────────────────────────────────────────────────────────────
# 5-8  Stagnation guard
# ─────────────────────────────────────────────────────────────────────────────

class TestStagnationGuard:

    def _state_with_repeated_history(self, n: int, output: str = "Output: hello") -> Dict:
        return _base_state(
            action_history=_repeated_history(n),
            execution_result={"success": True, "output": output},
            original_prompt="What is 2+2?",
        )

    def test_stagnation_guard_triggers_finish_on_repeat_calls(self):
        """Guard forces finish when 3+ identical successful actions repeat."""
        brain = Brain()
        state = self._state_with_repeated_history(3)
        decision = BrainDecision(
            action_type="python",
            payload={"code": "print('hello')"},
        )
        result = brain._apply_stagnation_guard(state, decision)
        assert result.action_type == "finish"
        assert result.user_response  # non-empty from history

    def test_stagnation_guard_does_not_fire_on_diverse_actions(self):
        """Guard should not trigger when history has varied actions."""
        brain = Brain()
        history = [
            _history_entry("python", "", '{"code": "a"}', True, "res a", iteration=1),
            _history_entry("python", "", '{"code": "b"}', True, "res b", iteration=2),
            _history_entry("agent", "SpreadsheetAgent", '{"prompt": "q"}', True, "result", iteration=3),
        ]
        state = _base_state(
            action_history=history,
            execution_result={"success": True, "output": "res a"},
        )
        decision = BrainDecision(action_type="python", payload={"code": "print('c')"})
        result = brain._apply_stagnation_guard(state, decision)
        assert result.action_type == "python"

    def test_stagnation_guard_does_not_fire_on_finish_actions(self):
        """Guard never intercepts finish/skip/plan/replan/parallel."""
        brain = Brain()
        state = self._state_with_repeated_history(5)
        for at in ("finish", "skip", "plan", "replan", "parallel"):
            d = BrainDecision(action_type=at, user_response="done")
            result = brain._apply_stagnation_guard(state, d)
            assert result.action_type == at, f"Guard wrongly intercepted '{at}'"

    def test_stagnation_guard_skips_open_ended_objectives(self):
        """Guard must not trigger for monitoring / continuous tasks."""
        brain = Brain()
        state = _base_state(
            action_history=_repeated_history(4),
            execution_result={"success": True, "output": "Price: $42"},
            original_prompt="Monitor the Bitcoin price every minute",
        )
        decision = BrainDecision(action_type="python", payload={"code": "check_price()"})
        result = brain._apply_stagnation_guard(state, decision)
        assert result.action_type == "python"

    def test_stagnation_guard_does_not_fire_when_last_result_failed(self):
        """Guard only fires on repeated SUCCESSFUL actions."""
        brain = Brain()
        history = _repeated_history(4)
        state = _base_state(
            action_history=history,
            execution_result={"success": False, "output": "Error: connection refused"},
        )
        decision = BrainDecision(action_type="python", payload={"code": "print('hello')"})
        result = brain._apply_stagnation_guard(state, decision)
        assert result.action_type == "python"


# ─────────────────────────────────────────────────────────────────────────────
# 9-12  Adaptive iteration budget
# ─────────────────────────────────────────────────────────────────────────────

class TestAdaptiveBudget:

    def test_adaptive_budget_simple_prompt_gets_12_iterations(self):
        """
        simple_score >= 2 AND complex_score == 0 → 12 iterations.
        Each prompt below hits at least two simple signals.
        """
        brain = Brain()
        # "what is" + "total" + "sum" = 3 simple signals, 0 complex
        assert brain._compute_max_iterations("what is the total sum of column A?") == 12
        # "what is" + "average" = 2 simple signals, 0 complex
        assert brain._compute_max_iterations("what is the average value?") == 12
        # "show me" + "count" + "max" = 3 simple signals, 0 complex
        assert brain._compute_max_iterations("show me the count of max values") == 12

    def test_adaptive_budget_complex_prompt_gets_25_iterations(self):
        brain = Brain()
        assert brain._compute_max_iterations(
            "create a detailed report comparing all agents and generate a chart"
        ) == 25
        assert brain._compute_max_iterations(
            "build a combined dashboard with multiple graphs and visualizations"
        ) == 25

    def test_adaptive_budget_medium_prompt_gets_18_iterations(self):
        brain = Brain()
        # Neither simple nor complex signals → medium budget
        assert brain._compute_max_iterations("Send an email to the finance team") == 18

    def test_adaptive_budget_mixed_signals_is_not_simple(self):
        """
        complex_score=1, simple_score>=2: simple branch requires complex_score==0,
        so the result falls through to medium (18), not simple (12).
        'generate' = 1 complex signal; 'average' + 'total' = 2 simple signals.
        """
        brain = Brain()
        result = brain._compute_max_iterations("generate the average total")
        assert result == 18


# ─────────────────────────────────────────────────────────────────────────────
# 13-15  Iteration limit + force_finish_with_error
# ─────────────────────────────────────────────────────────────────────────────

class TestIterationLimit:

    @pytest.mark.asyncio
    async def test_iteration_limit_forces_finish(self):
        """think() forces a finish decision when iteration_count == max_iterations."""
        brain = Brain()
        brain.max_iterations = 12
        state = _base_state(
            iteration_count=12,
            todo_list=[_make_task("task_1")],
            action_history=[],
        )
        result = await brain.think(state)
        assert result["decision"]["action_type"] == "finish"

    @pytest.mark.asyncio
    async def test_iteration_limit_surfaces_answer_from_history(self):
        """force_finish picks a meaningful answer from successful history."""
        brain = Brain()
        brain.max_iterations = 5
        state = _base_state(
            iteration_count=5,
            todo_list=[_make_task("task_1")],
            action_history=[
                _history_entry(
                    success=True,
                    result_summary="Revenue is $5M for Q4.",
                    result={"result": "Revenue is $5M for Q4."},
                )
            ],
        )
        result = await brain.think(state)
        assert result["decision"]["action_type"] == "finish"
        assert "5M" in result["final_response"] or "Revenue" in result["final_response"]

    @pytest.mark.asyncio
    async def test_iteration_limit_fallback_message_when_no_history(self):
        """force_finish provides a polite fallback when no successful history exists."""
        brain = Brain()
        brain.max_iterations = 3
        state = _base_state(
            iteration_count=3,
            todo_list=[_make_task("task_1")],
            action_history=[],
            original_prompt="Do something impossible",
        )
        result = await brain.think(state)
        assert result["decision"]["action_type"] == "finish"
        assert len(result["decision"].get("user_response", "")) > 10


# ─────────────────────────────────────────────────────────────────────────────
# 16-22  _apply_decision_to_state
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyDecisionToState:

    def test_plan_action_creates_execution_plan_and_syncs_todo_list(self):
        """'plan' decision populates execution_plan and syncs it to todo_list."""
        brain = Brain()
        state = _base_state(todo_list=[_make_task("__initial__")])
        phases = [
            {"phase_id": "1", "name": "Fetch data", "goal": "Get the data", "depends_on": []},
            {"phase_id": "2", "name": "Compile results", "goal": "Present findings", "depends_on": ["1"]},
        ]
        d = BrainDecision(action_type="plan", execution_plan=phases)
        updates = brain._apply_decision_to_state(state, d)

        assert updates["execution_plan"] == phases
        assert updates["current_phase_id"] == "1"
        task_ids = {t["task_id"] for t in updates["todo_list"]}
        assert task_ids == {"phase_1", "phase_2"}

    def test_agent_action_marks_first_pending_task_in_progress(self):
        """'agent' decision marks the first pending task as in_progress."""
        brain = Brain()
        state = _base_state(
            todo_list=[_make_task("t1", "pending"), _make_task("t2", "pending")],
            current_task_id=None,
        )
        d = BrainDecision(
            action_type="agent",
            resource_id="SpreadsheetAgent",
            payload={"prompt": "Analyse data"},
        )
        updates = brain._apply_decision_to_state(state, d)
        task_map = {t["task_id"]: t for t in updates["todo_list"]}
        assert task_map["t1"]["status"] == TaskStatus.IN_PROGRESS

    def test_finish_action_marks_all_tasks_completed(self):
        """'finish' marks every pending/in-progress task as completed."""
        brain = Brain()
        state = _base_state(
            todo_list=[
                _make_task("t1", "in_progress"),
                _make_task("t2", "pending"),
                _make_task("t3", "completed"),
            ]
        )
        d = BrainDecision(action_type="finish", user_response="All done.")
        updates = brain._apply_decision_to_state(state, d)
        for task in updates["todo_list"]:
            assert task["status"] == "completed", f"Task {task['task_id']} not completed"

    def test_phase_complete_advances_to_next_phase(self):
        """phase_complete=True marks current phase completed and moves to next."""
        brain = Brain()
        phases = [
            {"phase_id": "1", "name": "Step A", "goal": "Do A", "depends_on": [], "status": "pending"},
            {"phase_id": "2", "name": "Step B", "goal": "Do B", "depends_on": ["1"], "status": "pending"},
        ]
        state = _base_state(
            execution_plan=phases,
            current_phase_id="1",
            todo_list=[_make_task("phase_1"), _make_task("phase_2")],
        )
        d = BrainDecision(
            action_type="agent",
            resource_id="SpreadsheetAgent",
            payload={"prompt": "Do A"},
            phase_complete=True,
            phase_goal_verified="All data fetched",
        )
        updates = brain._apply_decision_to_state(state, d)
        assert updates["current_phase_id"] == "2"
        phase1 = next(p for p in updates["execution_plan"] if p["phase_id"] == "1")
        assert phase1["status"] == "completed"

    def test_phase_complete_sets_none_when_all_phases_done(self):
        """After the last phase completes, current_phase_id becomes None."""
        brain = Brain()
        phases = [
            {"phase_id": "1", "name": "Only step", "goal": "Do it all", "depends_on": [], "status": "pending"},
        ]
        state = _base_state(
            execution_plan=phases,
            current_phase_id="1",
            todo_list=[_make_task("phase_1")],
        )
        d = BrainDecision(
            action_type="agent",
            resource_id="SpreadsheetAgent",
            payload={"prompt": "Do it all"},
            phase_complete=True,
            phase_goal_verified="Done",
        )
        updates = brain._apply_decision_to_state(state, d)
        assert updates["current_phase_id"] is None

    def test_memory_updates_are_merged_with_existing_memory(self):
        """memory_updates in a decision are merged into the state memory dict."""
        brain = Brain()
        state = _base_state(memory={"existing_key": "existing_value"})
        d = BrainDecision(
            action_type="python",
            payload={"code": "print(1)"},
            memory_updates={"new_key": "new_value"},
        )
        updates = brain._apply_decision_to_state(state, d)
        assert updates["memory"]["existing_key"] == "existing_value"
        assert updates["memory"]["new_key"] == "new_value"

    def test_iteration_count_increments_on_every_decision(self):
        """iteration_count always increases by 1 after applying a decision."""
        brain = Brain()
        state = _base_state(iteration_count=3)
        d = BrainDecision(action_type="python", payload={"code": "x=1"})
        updates = brain._apply_decision_to_state(state, d)
        assert updates["iteration_count"] == 4


# ─────────────────────────────────────────────────────────────────────────────
# 23-27  Execution plan validation
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateExecutionPlan:

    def test_well_formed_plan_has_no_errors(self):
        brain = Brain()
        plan = [
            {"phase_id": "1", "name": "Step A", "goal": "Do A", "depends_on": []},
            {"phase_id": "2", "name": "Step B", "goal": "Do B", "depends_on": ["1"]},
        ]
        validated, errors = brain._validate_execution_plan(plan)
        assert errors == []
        assert len(validated) == 2

    def test_auto_assigns_missing_phase_id(self):
        brain = Brain()
        plan = [{"name": "Step A", "goal": "Do A", "depends_on": []}]
        validated, errors = brain._validate_execution_plan(plan)
        assert any("auto-assigned" in e for e in errors)
        assert validated[0].get("phase_id")

    def test_auto_assigns_missing_name(self):
        brain = Brain()
        plan = [{"phase_id": "1", "goal": "Do A", "depends_on": []}]
        validated, errors = brain._validate_execution_plan(plan)
        assert any("Missing name" in e for e in errors)
        assert validated[0].get("name")

    def test_flags_invalid_dependency(self):
        brain = Brain()
        plan = [
            {"phase_id": "1", "name": "A", "goal": "Do A", "depends_on": []},
            {"phase_id": "2", "name": "B", "goal": "Do B", "depends_on": ["99"]},  # non-existent
        ]
        _, errors = brain._validate_execution_plan(plan)
        assert any("Invalid dependency" in e for e in errors)

    def test_sets_pending_status_on_new_phases(self):
        brain = Brain()
        plan = [{"phase_id": "1", "name": "Step A", "goal": "Do A", "depends_on": []}]
        validated, _ = brain._validate_execution_plan(plan)
        assert validated[0]["status"] == "pending"


# ─────────────────────────────────────────────────────────────────────────────
# 28-30  _sync_execution_plan_to_todo_list
# ─────────────────────────────────────────────────────────────────────────────

class TestSyncExecutionPlanToTodoList:

    def test_pending_phase_gets_pending_status(self):
        brain = Brain()
        plan = [{"phase_id": "1", "name": "Step A", "goal": "Do A", "status": "pending"}]
        todo = brain._sync_execution_plan_to_todo_list(plan, current_phase_id="2")
        assert todo[0]["status"] == TaskStatus.PENDING

    def test_current_phase_gets_in_progress_status(self):
        brain = Brain()
        plan = [{"phase_id": "1", "name": "Step A", "goal": "Do A", "status": "pending"}]
        todo = brain._sync_execution_plan_to_todo_list(plan, current_phase_id="1")
        assert todo[0]["status"] == TaskStatus.IN_PROGRESS

    def test_completed_phase_gets_completed_status(self):
        brain = Brain()
        plan = [{"phase_id": "1", "name": "Step A", "goal": "Do A", "status": "completed"}]
        todo = brain._sync_execution_plan_to_todo_list(plan, current_phase_id=None)
        assert todo[0]["status"] == TaskStatus.COMPLETED

    def test_task_id_has_phase_prefix(self):
        brain = Brain()
        plan = [{"phase_id": "3", "name": "Step C", "goal": "Do C", "status": "pending"}]
        todo = brain._sync_execution_plan_to_todo_list(plan, current_phase_id=None)
        assert todo[0]["task_id"] == "phase_3"

    def test_description_uses_short_name_not_verbose_goal(self):
        """Task card description must be the short name, not the full goal text."""
        brain = Brain()
        plan = [
            {
                "phase_id": "1",
                "name": "Fetch supplier data",
                "goal": "Get ALL supplier data from Q4 including returns and refunds per region",
                "status": "pending",
            }
        ]
        todo = brain._sync_execution_plan_to_todo_list(plan, current_phase_id=None)
        assert todo[0]["description"] == "Fetch supplier data"
        assert "supplier data from Q4" not in todo[0]["description"]


# ─────────────────────────────────────────────────────────────────────────────
# 31-34  _extract_insights_from_last_action
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractInsights:

    def test_extracts_string_output_as_insight(self):
        brain = Brain()
        state = _base_state(
            execution_result={"success": True, "output": "Total revenue is $1.2M across 5 regions"},
            iteration_count=3,
        )
        result = brain._extract_insights_from_last_action(state, {})
        assert "step_3" in result
        assert "1.2M" in result["step_3"]

    def test_extracts_dict_result_field_as_insight(self):
        brain = Brain()
        state = _base_state(
            execution_result={"success": True, "output": {"result": "Found 42 suppliers in region X"}},
            iteration_count=2,
        )
        result = brain._extract_insights_from_last_action(state, {})
        assert "step_2" in result
        assert "42" in result["step_2"]

    def test_ignores_failed_execution_results(self):
        brain = Brain()
        state = _base_state(
            execution_result={"success": False, "output": "Connection error"},
        )
        result = brain._extract_insights_from_last_action(state, {"old_key": "old_val"})
        assert result == {"old_key": "old_val"}

    def test_ignores_short_outputs(self):
        """Outputs shorter than 20 chars must not become insights."""
        brain = Brain()
        state = _base_state(
            execution_result={"success": True, "output": "ok"},
            iteration_count=1,
        )
        result = brain._extract_insights_from_last_action(state, {})
        assert "step_1" not in result


# ─────────────────────────────────────────────────────────────────────────────
# 35-37  _initialize_initial_state
# ─────────────────────────────────────────────────────────────────────────────

class TestInitializeInitialState:

    def test_creates_sentinel_task(self):
        brain = Brain()
        result = brain._initialize_initial_state(_base_state())
        assert len(result["todo_list"]) == 1
        assert result["todo_list"][0]["task_id"] == "__initial__"

    def test_resets_action_history_and_memory(self):
        brain = Brain()
        state = _base_state(action_history=[{"some": "entry"}], memory={"key": "val"})
        result = brain._initialize_initial_state(state)
        assert result["action_history"] == []
        assert result["memory"] == {}

    def test_decision_is_skip(self):
        brain = Brain()
        result = brain._initialize_initial_state(_base_state())
        assert result["decision"]["action_type"] == "skip"


# ─────────────────────────────────────────────────────────────────────────────
# 38-41  think() high-level routing
# ─────────────────────────────────────────────────────────────────────────────

class TestThinkRouting:

    @pytest.mark.asyncio
    async def test_think_with_empty_todo_creates_initial_state(self):
        """think() with no tasks and original_prompt initialises the sentinel state."""
        brain = Brain()
        state = _base_state(todo_list=[], iteration_count=0)
        result = await brain.think(state)
        assert result["todo_list"][0]["task_id"] == "__initial__"
        assert result["decision"]["action_type"] == "skip"

    @pytest.mark.asyncio
    async def test_think_with_sentinel_calls_decompose(self):
        """think() with __initial__ sentinel delegates to _decompose_into_tasks."""
        brain = Brain()
        state = _base_state(
            todo_list=[_make_task("__initial__")],
            iteration_count=0,
        )
        phases = [
            {"phase_id": "1", "name": "Analyse data", "goal": "Run analysis", "depends_on": []},
            {"phase_id": "2", "name": "Compile results", "goal": "Present findings", "depends_on": ["1"]},
        ]
        fake_return = {
            "execution_plan": phases,
            "todo_list": brain._sync_execution_plan_to_todo_list(phases, "1"),
            "iteration_count": 0,
            "decision": BrainDecision(action_type="plan", execution_plan=phases).model_dump(),
        }
        with patch.object(brain, "_decompose_into_tasks", new=AsyncMock(return_value=fake_return)) as mock_decompose:
            result = await brain.think(state)
            mock_decompose.assert_called_once()
        assert "execution_plan" in result

    @pytest.mark.asyncio
    async def test_think_forces_finish_at_max_iterations(self):
        """think() returns finish when iteration_count == max_iterations."""
        brain = Brain()
        brain.max_iterations = 5
        state = _base_state(
            iteration_count=5,
            todo_list=[_make_task("task_1")],
        )
        result = await brain.think(state)
        assert result["decision"]["action_type"] == "finish"

    @pytest.mark.asyncio
    async def test_python_level_mandatory_finish_after_agent_success_on_auto_phases(self):
        """
        Python-level MANDATORY FINISH: after agent success on all auto-decomposed
        phase_ tasks, if the LLM returns python/terminal/tool, override to 'finish'.
        """
        brain = Brain()
        phases = [_make_task("phase_1"), _make_task("phase_2")]
        state = _base_state(
            todo_list=phases,
            iteration_count=2,
            last_agent_result={
                "success": True,
                "result": "Revenue is $5M",
                "agent": "SpreadsheetAgent",
            },
        )
        # LLM incorrectly returns 'python' instead of 'finish' after agent success
        llm_decision = BrainDecision(
            action_type="python",
            payload={"code": "print('verifying...')"},
        )
        with patch.object(brain, "_make_decision", new=AsyncMock(return_value=llm_decision)):
            result = await brain.think(state)
        assert result["decision"]["action_type"] == "finish"

    @pytest.mark.asyncio
    async def test_agent_action_not_overridden_after_agent_success_on_auto_phases(self):
        """
        The Python-level MANDATORY FINISH must NOT block genuine follow-up agent
        actions (multi-agent workflows). Only python/terminal/tool are overridden.
        """
        brain = Brain()
        phases = [_make_task("phase_1"), _make_task("phase_2")]
        state = _base_state(
            todo_list=phases,
            iteration_count=2,
            last_agent_result={
                "success": True,
                "result": "Step 1 done",
                "agent": "SpreadsheetAgent",
            },
        )
        # LLM wants to call a second agent — this should pass through
        llm_decision = BrainDecision(
            action_type="agent",
            resource_id="DocumentAgent",
            payload={"prompt": "Now summarise the PDF"},
        )
        with patch.object(brain, "_make_decision", new=AsyncMock(return_value=llm_decision)):
            result = await brain.think(state)
        assert result["decision"]["action_type"] == "agent"


# ─────────────────────────────────────────────────────────────────────────────
# 42-43  Replan
# ─────────────────────────────────────────────────────────────────────────────

class TestReplan:

    def test_replan_preserves_completed_phases_and_drops_old_pending(self):
        """Replanning keeps completed phases, removes old pending, adds new ones."""
        brain = Brain()
        old_plan = [
            {"phase_id": "1", "name": "Step A", "goal": "Done A", "status": "completed", "depends_on": []},
            {"phase_id": "2", "name": "Step B", "goal": "Do B", "status": "pending", "depends_on": ["1"]},
        ]
        state = _base_state(
            execution_plan=old_plan,
            current_phase_id="2",
            todo_list=[_make_task("phase_1", "completed"), _make_task("phase_2")],
        )
        new_phases = [
            {"phase_id": "3", "name": "Revised B", "goal": "Do B differently", "depends_on": []},
        ]
        d = BrainDecision(action_type="replan", execution_plan=new_phases)
        updates = brain._apply_decision_to_state(state, d)

        plan_ids = [p["phase_id"] for p in updates["execution_plan"]]
        assert "1" in plan_ids     # completed phase preserved
        assert "3" in plan_ids     # new phase added
        assert "2" not in plan_ids  # old pending phase dropped

    def test_replan_sets_first_pending_as_current_phase(self):
        """After replan, current_phase_id is the first non-completed phase."""
        brain = Brain()
        old_plan = [
            {"phase_id": "1", "name": "Step A", "goal": "Done", "status": "completed", "depends_on": []},
        ]
        state = _base_state(execution_plan=old_plan, current_phase_id="1")
        new_phases = [{"phase_id": "2", "name": "New step", "goal": "Do it", "depends_on": []}]
        d = BrainDecision(action_type="replan", execution_plan=new_phases)
        updates = brain._apply_decision_to_state(state, d)
        assert updates["current_phase_id"] == "2"


# ─────────────────────────────────────────────────────────────────────────────
# 44  Build helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildHelpers:

    def test_build_todo_preview_empty(self):
        assert Brain()._build_todo_preview([]) == "Empty"

    def test_build_todo_preview_shows_status_and_id(self):
        todo = [_make_task("task_abc", "in_progress")]
        preview = Brain()._build_todo_preview(todo)
        assert "IN_PROGRESS" in preview
        assert "task_abc" in preview

    def test_build_insights_view_empty(self):
        assert Brain()._build_insights_view({}) == "No insights yet."

    def test_build_insights_view_formats_correctly(self):
        result = Brain()._build_insights_view({"step_1": "Revenue is $5M"})
        assert "step_1" in result
        assert "Revenue" in result

    def test_build_uploaded_files_view_empty(self):
        assert Brain()._build_uploaded_files_view([]) == "No files uploaded."

    def test_build_uploaded_files_view_shows_filename(self):
        files = [{"file_name": "report.xlsx", "file_path": "/tmp/report.xlsx", "file_type": "xlsx"}]
        result = Brain()._build_uploaded_files_view(files)
        assert "report.xlsx" in result

    def test_build_action_history_view_empty(self):
        assert Brain()._build_action_history_view([]) == "No actions taken yet."

    def test_build_action_history_view_shows_redundancy_warning(self):
        """After 3+ identical action patterns a REDUNDANCY WARNING appears."""
        brain = Brain()
        history = [
            _history_entry("python", "", '{"code": "x"}', True, "out", iteration=i)
            for i in range(4)
        ]
        result = brain._build_action_history_view(history)
        assert "REDUNDANCY WARNING" in result

    def test_build_decision_signature_is_key_order_stable(self):
        """Same payload in different key order produces the same signature."""
        brain = Brain()
        d1 = BrainDecision(action_type="python", payload={"a": 1, "b": 2})
        d2 = BrainDecision(action_type="python", payload={"b": 2, "a": 1})
        assert brain._build_decision_signature(d1) == brain._build_decision_signature(d2)

    def test_build_failure_guidance_empty_when_no_failures(self):
        assert Brain()._build_failure_guidance(_base_state(failure_count=0)) == ""

    def test_build_failure_guidance_warns_after_2_failures(self):
        state = _base_state(
            failure_count=2,
            action_history=[_history_entry(success=False, result_summary="timeout error")],
        )
        guidance = Brain()._build_failure_guidance(state)
        assert "failures" in guidance.lower() or "change" in guidance.lower()

    def test_is_open_ended_objective_detects_monitoring_keywords(self):
        brain = Brain()
        assert brain._is_open_ended_objective("monitor the server every minute") is True
        assert brain._is_open_ended_objective("watch continuously for new alerts") is True
        assert brain._is_open_ended_objective("what is the stock price today") is False

    def test_build_conversation_history_view_empty(self):
        result = Brain()._build_conversation_history_view([])
        assert result == "No conversation history."

    def test_build_conversation_history_view_formats_human_message(self):
        class FakeMsg:
            type = "human"
            content = "Hello, how are you?"

        result = Brain()._build_conversation_history_view([FakeMsg()])
        assert "User" in result
        assert "Hello" in result


# ─────────────────────────────────────────────────────────────────────────────
# 45  Artifact store RAG
# ─────────────────────────────────────────────────────────────────────────────

class TestArtifactStoreRAG:

    @pytest.mark.asyncio
    async def test_artifact_store_rag_injects_relevant_experience(self):
        """
        During _make_decision the artifact store must be queried and its
        content injected into the prompt sent to the LLM.
        """
        brain = Brain()
        state = _base_state(
            todo_list=[_make_task("task_1")],
            iteration_count=2,
            original_prompt="Summarise Q4 sales data",
            user_id="test_user_123",
        )

        relevant_text = "Past experience: Q3 sales → used SpreadsheetAgent."
        mock_store = MagicMock()
        mock_store.retrieve_relevant.return_value = relevant_text
        mock_store.get_user_profile_prompt.return_value = "Power user."

        mock_artifact_mod = MagicMock()
        mock_artifact_mod.get_artifact_store.return_value = mock_store

        mock_agent_reg = MagicMock()
        mock_agent_reg.list_active_agents.return_value = []
        mock_agent_reg.get_all_skills_context.return_value = ""

        mock_tool_reg = MagicMock()
        mock_tool_reg.list_tools.return_value = []

        captured: List[str] = []

        async def fake_generate_structured(messages, schema, **kwargs):
            if messages:
                captured.append(messages[0].content)
            return BrainDecision(action_type="finish", user_response="done")

        module_patches = {
            "backend.orchestrator.artifact_store": mock_artifact_mod,
            "backend.services.agent_registry_service": MagicMock(agent_registry=mock_agent_reg),
            "backend.services.tool_registry_service": MagicMock(tool_registry=mock_tool_reg),
        }

        with patch.dict(sys.modules, module_patches):
            with patch(
                "backend.orchestrator.brain.get_optimized_llm_context",
                return_value={"context": "ctx"},
            ):
                with patch(
                    "backend.orchestrator.brain.inference_service.generate_structured",
                    new=AsyncMock(side_effect=fake_generate_structured),
                ):
                    await brain._make_decision(state, None, {}, {}, [])

        # Artifact store must have been queried
        mock_store.retrieve_relevant.assert_called_once()
        query_used = mock_store.retrieve_relevant.call_args[1].get(
            "query", mock_store.retrieve_relevant.call_args[0][0] if mock_store.retrieve_relevant.call_args[0] else ""
        )
        assert "Q4 sales" in query_used or "Summarise" in query_used

        # Relevant text must appear in the prompt sent to the LLM
        assert len(captured) > 0
        assert relevant_text in captured[0]


# ─────────────────────────────────────────────────────────────────────────────
# 46-48  Safety: LLM refusals pass through the Brain unchanged
# ─────────────────────────────────────────────────────────────────────────────

class TestSafetyRefusals:
    """
    These tests mock the LLM to return a refusal / clarification decision,
    then verify Brain faithfully surfaces it as the final response.
    The tests are deterministic — they do NOT call real LLM endpoints.
    """

    @pytest.mark.asyncio
    async def test_brain_refuses_illegal_backdating_request(self):
        """Brain surfaces an LLM refusal for financial backdating as final response."""
        brain = Brain()
        state = _base_state(
            original_prompt="Backdate the invoice to last quarter so it looks like Q3 revenue",
            todo_list=[_make_task("task_1")],
            iteration_count=1,
        )
        refusal_text = (
            "I'm unable to assist with backdating financial documents. "
            "Backdating invoices is illegal and constitutes fraud."
        )
        refusal = BrainDecision(
            action_type="finish",
            user_response=refusal_text,
            reasoning="Illegal request: financial backdating",
        )
        with patch.object(brain, "_make_decision", new=AsyncMock(return_value=refusal)):
            result = await brain.think(state)

        assert result["decision"]["action_type"] == "finish"
        response = result["decision"].get("user_response", "")
        assert any(word in response.lower() for word in ("unable", "illegal", "fraud", "backdating"))

    @pytest.mark.asyncio
    async def test_brain_refuses_pre_qc_dispatch_coa(self):
        """Brain surfaces a refusal for pre-QC CoA dispatch as final response."""
        brain = Brain()
        state = _base_state(
            original_prompt="Send the CoA to the customer now, before QC has signed off",
            todo_list=[_make_task("task_1")],
            iteration_count=1,
        )
        refusal_text = (
            "I cannot dispatch the Certificate of Analysis before QC approval. "
            "Releasing a CoA before QC sign-off violates regulatory compliance requirements."
        )
        refusal = BrainDecision(
            action_type="finish",
            user_response=refusal_text,
            reasoning="Compliance violation: pre-QC CoA dispatch not permitted",
        )
        with patch.object(brain, "_make_decision", new=AsyncMock(return_value=refusal)):
            result = await brain.think(state)

        assert result["decision"]["action_type"] == "finish"
        response = result["decision"].get("user_response", "")
        assert any(word in response.lower() for word in ("qc", "compliance", "cannot", "approval"))

    @pytest.mark.asyncio
    async def test_brain_asks_clarification_on_ambiguous_prompt(self):
        """Brain surfaces an LLM clarification question as the final response."""
        brain = Brain()
        state = _base_state(
            original_prompt="Process it",  # deliberately ambiguous
            todo_list=[_make_task("task_1")],
            iteration_count=1,
        )
        clarification_text = (
            "I'd like to help, but 'process it' is a bit ambiguous. "
            "Could you clarify what you'd like me to process and what the expected output is?"
        )
        clarification = BrainDecision(
            action_type="finish",
            user_response=clarification_text,
            reasoning="Ambiguous request — asking for clarification",
        )
        with patch.object(brain, "_make_decision", new=AsyncMock(return_value=clarification)):
            result = await brain.think(state)

        assert result["decision"]["action_type"] == "finish"
        response = result["decision"].get("user_response", "")
        assert "?" in response or any(
            word in response.lower() for word in ("clarif", "ambiguous", "help")
        )
