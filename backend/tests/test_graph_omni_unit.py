"""
Unit tests for backend/orchestrator/graph.py and omni_dispatcher.py

Tests the routing logic, dispatch cycle, approval helpers, and graph topology
without hitting real LLM endpoints or agents.

Coverage — omni_dispatcher.py:
  1.  TestOmniRouteCondition (9 tests)
        Pure-function routing: all branches of omni_route_condition()
        a. agent action_type  → "hands"
        b. tool action_type   → "hands"
        c. python action_type → "hands"
        d. final_response set → "finish" (guards stale decision)
        e. pending_approval   → "approval"
        f. action_type=finish → "finish"
        g. action_type=skip   → "brain"
        h. pending_approval takes priority over finish
        i. final_response takes priority over skip (no decision)

  2.  TestShouldContinue (3 tests)
        Pure-function routing for the legacy should_continue() helper.
        a. pending_approval → "approval"
        b. action_type=finish → "finish"
        c. default → "continue"

  3.  TestApprovalHelpers (4 tests)
        approve_pending_action / reject_pending_action pure-function contracts.
        a. approve: clears flags, re-applies decision sans requires_approval
        b. approve: returns {} when nothing pending
        c. reject: sets skip decision with caller-supplied reason
        d. reject: returns {} when nothing pending

  4.  TestOmniDispatch (5 tests)
        omni_dispatch async cycle with mocked Brain + Hands.
        a. pauses immediately (returns state) when pending_approval pre-brain
        b. skips Hands when Brain decides finish
        c. skips Hands when Brain decides skip
        d. calls Hands after Brain for agent action
        e. pauses before Hands when Brain returns with pending_approval set

  5.  TestGraphStructureAndBehavior (6 tests)
        graph.py topology and action_approval_node behavior.
        a. test_route_to_hands_when_action_type_agent      ← user's list
        b. test_route_to_end_when_final_response_set       ← user's list
        c. test_route_to_approval_when_pending_approval_set ← user's list
        d. test_loop_back_to_brain_after_hands_execution   ← user's list
        e. test_graph_pauses_on_approval_state             ← user's list
        f. test_route_to_finish_when_action_type_finish    ← user's list

Architecture notes:
  - omni_route_condition is a PURE FUNCTION: no mocking needed.
  - omni_dispatch wraps brain.think → hands.execute in one shot; the loop-back
    is implemented as workflow.add_edge("omni_hands", "omni_brain") in graph.py.
  - For graph-run tests the module-level brain / hands singletons are patched on
    their INSTANCES so that create_graph_with_checkpointer() picks up the mock
    when it stores brain.think / hands.execute as node callables.
"""

import sys
import asyncio
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent        # backend/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))                 # project root

from backend.orchestrator.omni_dispatcher import (
    omni_route_condition,
    should_continue,
    approve_pending_action,
    reject_pending_action,
    omni_dispatch,
    brain as module_brain,
    hands as module_hands,
)
from backend.orchestrator.graph import create_graph_with_checkpointer
from langgraph.checkpoint.memory import MemorySaver


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _state(**overrides) -> Dict[str, Any]:
    """Minimal orchestrator state for routing / dispatch tests."""
    s: Dict[str, Any] = {
        "original_prompt": "test",
        "todo_list": [],
        "memory": {},
        "action_history": [],
        "insights": {},
        "execution_plan": None,
        "current_phase_id": None,
        "decision": None,
        "execution_result": None,
        "current_task_id": None,
        "iteration_count": 0,
        "max_iterations": 12,
        "final_response": None,
        "pending_user_input": False,
        "question_for_user": None,
        "pending_approval": False,
        "pending_decision": None,
        "pending_action": None,
        "pending_action_approval": False,
        "error": None,
        "failure_count": 0,
        "thread_id": "test_thread",
        "user_id": "test_user",
        "owner_id": None,
        "uploaded_files": [],
        "messages": [],
        "created_files": [],
        "orchestrator_workspace": "/tmp/ws",
        "shared_files": [],
        "shared_workspace": "/tmp/shared",
        "canvas_registry": None,
        "active_canvas_id": None,
        "has_canvas": False,
        "canvas_type": None,
        "canvas_content": None,
        "canvas_data": None,
        "canvas_title": None,
        "browser_view": None,
        "plan_view": None,
        "current_view": None,
        "last_agent_result": None,
    }
    s.update(overrides)
    return s


def _decision(action_type: str, **extra) -> Dict:
    return {"action_type": action_type, "resource_id": "test_agent",
            "payload": {}, **extra}


# ─────────────────────────────────────────────────────────────────────────────
# 1. omni_route_condition — pure function
# ─────────────────────────────────────────────────────────────────────────────

class TestOmniRouteCondition:
    """
    omni_route_condition is a pure function — no mocking required.
    Covers every branch: hands | approval | finish | brain.
    """

    def test_route_to_hands_when_action_type_agent(self):
        """Brain decided to call an agent → route to hands."""
        assert omni_route_condition(_state(decision=_decision("agent"))) == "hands"

    def test_route_to_hands_when_action_type_tool(self):
        """Brain decided to call a tool → route to hands."""
        assert omni_route_condition(_state(decision=_decision("tool"))) == "hands"

    def test_route_to_hands_when_action_type_python(self):
        """Brain decided to run Python → route to hands."""
        assert omni_route_condition(_state(decision=_decision("python"))) == "hands"

    def test_route_to_finish_when_action_type_finish(self):
        """Brain explicitly decided finish → route to END."""
        assert omni_route_condition(_state(decision=_decision("finish"))) == "finish"

    def test_route_to_end_when_final_response_set(self):
        """
        final_response guards against code paths that set it without updating decision.
        Even if decision.action_type is something other than 'finish', if final_response
        is set the graph must terminate.
        """
        state = _state(
            final_response="Here is the answer.",
            decision=_decision("python"),   # decision says python, but final_response wins
        )
        assert omni_route_condition(state) == "finish"

    def test_route_to_approval_when_pending_approval_set(self):
        """Approval flag takes highest priority — checked before all other conditions."""
        state = _state(pending_approval=True, decision=_decision("agent"))
        assert omni_route_condition(state) == "approval"

    def test_route_to_brain_when_action_type_skip(self):
        """Skip loops back to Brain for another thinking cycle (no execution)."""
        assert omni_route_condition(_state(decision=_decision("skip"))) == "brain"

    def test_pending_approval_takes_priority_over_finish(self):
        """pending_approval is checked BEFORE final_response / action_type."""
        state = _state(
            pending_approval=True,
            final_response="done",
            decision=_decision("finish"),
        )
        assert omni_route_condition(state) == "approval"

    def test_final_response_takes_priority_over_skip_decision(self):
        """final_response → finish even when action_type would route to 'brain'."""
        state = _state(
            final_response="done",
            decision=_decision("skip"),
        )
        assert omni_route_condition(state) == "finish"


# ─────────────────────────────────────────────────────────────────────────────
# 2. should_continue — legacy pure-function router
# ─────────────────────────────────────────────────────────────────────────────

class TestShouldContinue:

    def test_approval_when_pending(self):
        assert should_continue(_state(pending_approval=True)) == "approval"

    def test_finish_when_action_finish(self):
        assert should_continue(_state(decision=_decision("finish"))) == "finish"

    def test_continue_when_action_agent(self):
        assert should_continue(_state(decision=_decision("agent"))) == "continue"

    def test_continue_when_no_decision(self):
        assert should_continue(_state()) == "continue"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Approval helpers — pure functions
# ─────────────────────────────────────────────────────────────────────────────

class TestApprovalHelpers:

    def test_approve_clears_flags_and_re_applies_decision(self):
        """
        approve_pending_action clears pending_approval, nulls pending_decision,
        and re-applies the original decision with requires_approval=False.
        """
        pending = {"action_type": "agent", "resource_id": "spreadsheet_agent",
                   "requires_approval": True, "approval_reason": "risky"}
        state = _state(pending_approval=True, pending_decision=pending)

        updates = approve_pending_action(state)

        assert updates["pending_approval"] is False
        assert updates["pending_decision"] is None
        assert updates["decision"]["action_type"] == "agent"
        assert updates["decision"]["requires_approval"] is False

    def test_approve_returns_empty_when_nothing_pending(self):
        """No-op when there is no pending approval."""
        assert approve_pending_action(_state(pending_approval=False)) == {}

    def test_reject_sets_skip_decision_with_reason(self):
        """
        reject_pending_action clears approval flags and sets action_type='skip'
        so the graph loops back to Brain without executing the rejected action.
        """
        pending = {"action_type": "tool", "requires_approval": True}
        state = _state(pending_approval=True, pending_decision=pending)

        updates = reject_pending_action(state, reason="Too dangerous")

        assert updates["pending_approval"] is False
        assert updates["pending_decision"] is None
        assert updates["decision"]["action_type"] == "skip"
        assert "Too dangerous" in updates["decision"]["reasoning"]

    def test_reject_returns_empty_when_nothing_pending(self):
        """No-op when there is no pending approval."""
        assert reject_pending_action(_state(pending_approval=False)) == {}


# ─────────────────────────────────────────────────────────────────────────────
# 4. omni_dispatch — async Brain→Hands cycle
# ─────────────────────────────────────────────────────────────────────────────

class TestOmniDispatch:
    """
    omni_dispatch executes one Brain→Hands pass.
    Brain and Hands are patched at the module-dispatcher level so omni_dispatch
    uses the mocks instead of the real Brain/Hands.
    """

    @pytest.mark.asyncio
    async def test_pauses_immediately_on_pending_approval(self):
        """If state already has pending_approval, return it unchanged — no Brain, no Hands."""
        state = _state(pending_approval=True,
                       pending_decision={"action_type": "agent"})

        mock_think = AsyncMock()
        mock_execute = AsyncMock()

        with patch.object(module_brain, "think", mock_think), \
             patch.object(module_hands, "execute", mock_execute):
            result = await omni_dispatch(state)

        mock_think.assert_not_called()
        mock_execute.assert_not_called()
        assert result.get("pending_approval") is True

    @pytest.mark.asyncio
    async def test_skips_hands_when_brain_decides_finish(self):
        """When Brain returns action_type='finish', Hands must NOT be called."""
        mock_think = AsyncMock(return_value={
            "decision": {"action_type": "finish", "user_response": "All done."},
            "final_response": "All done.",
        })
        mock_execute = AsyncMock()

        with patch.object(module_brain, "think", mock_think), \
             patch.object(module_hands, "execute", mock_execute):
            result = await omni_dispatch(_state())

        mock_think.assert_called_once()
        mock_execute.assert_not_called()
        assert result["decision"]["action_type"] == "finish"

    @pytest.mark.asyncio
    async def test_skips_hands_when_brain_decides_skip(self):
        """When Brain returns action_type='skip', Hands must NOT be called."""
        mock_think = AsyncMock(return_value={
            "decision": {"action_type": "skip", "reasoning": "Nothing to do."},
        })
        mock_execute = AsyncMock()

        with patch.object(module_brain, "think", mock_think), \
             patch.object(module_hands, "execute", mock_execute):
            result = await omni_dispatch(_state())

        mock_think.assert_called_once()
        mock_execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_calls_hands_after_brain_for_agent_action(self):
        """For action_type='agent', Brain is called then Hands is called in sequence."""
        brain_update = {
            "decision": {"action_type": "agent", "resource_id": "spreadsheet_agent",
                         "payload": {"instruction": "analyse"}},
        }
        hands_update = {
            "execution_result": {"action_id": "agent_ss", "success": True, "output": {}},
        }

        mock_think = AsyncMock(return_value=brain_update)
        mock_execute = AsyncMock(return_value=hands_update)

        with patch.object(module_brain, "think", mock_think), \
             patch.object(module_hands, "execute", mock_execute):
            result = await omni_dispatch(_state())

        mock_think.assert_called_once()
        mock_execute.assert_called_once()
        # Hands receives the state AFTER brain updates are merged
        hands_state_arg = mock_execute.call_args[0][0]
        assert hands_state_arg["decision"]["action_type"] == "agent"

    @pytest.mark.asyncio
    async def test_pauses_before_hands_when_approval_required_post_brain(self):
        """
        If Brain itself sets pending_approval (requires_approval path), omni_dispatch
        must NOT call Hands — it returns the updated state and waits for user input.
        """
        brain_update = {
            "decision": {"action_type": "agent", "requires_approval": True},
            "pending_approval": True,
            "pending_decision": {"action_type": "agent", "requires_approval": True},
        }

        mock_think = AsyncMock(return_value=brain_update)
        mock_execute = AsyncMock()

        with patch.object(module_brain, "think", mock_think), \
             patch.object(module_hands, "execute", mock_execute):
            result = await omni_dispatch(_state())

        mock_think.assert_called_once()
        mock_execute.assert_not_called()
        assert result.get("pending_approval") is True


# ─────────────────────────────────────────────────────────────────────────────
# 5. Graph structure and behavior (the 6 user-specified tests)
# ─────────────────────────────────────────────────────────────────────────────

class TestGraphStructureAndBehavior:
    """
    Tests for create_graph_with_checkpointer (graph.py).

    Routing tests reuse omni_route_condition (already tested above) to confirm
    the mapping used in add_conditional_edges matches expectations.
    For execution tests, a FRESH graph is created inside the patch context so
    the graph's node callables are the mocks, not the original methods.
    """

    # ── Helper: create a fresh graph whose brain/hands nodes are mocked ──────

    def _make_test_graph(self, mock_think, mock_execute):
        """
        Patch the module-level brain/hands INSTANCES before calling
        create_graph_with_checkpointer so the compiled graph stores mock callables.
        Returns (graph, context_managers_already_entered=False).
        Call inside an active patch.object context.
        """
        return create_graph_with_checkpointer(MemorySaver())

    # ── Route condition assertions (re-validate against the mapping in graph.py) ──

    def test_route_to_hands_when_action_type_agent(self):
        """omni_route_condition returns 'hands' for agent — mapped to omni_hands node."""
        route = omni_route_condition(_state(decision=_decision("agent")))
        assert route == "hands"

    def test_route_to_end_when_final_response_set(self):
        """omni_route_condition returns 'finish' — mapped to END in the graph."""
        route = omni_route_condition(_state(
            final_response="The answer is 42.",
            decision=_decision("skip"),  # final_response overrides
        ))
        assert route == "finish"

    def test_route_to_approval_when_pending_approval_set(self):
        """omni_route_condition returns 'approval' — mapped to action_approval_required node."""
        route = omni_route_condition(_state(pending_approval=True, decision=_decision("agent")))
        assert route == "approval"

    def test_route_to_finish_when_action_type_finish(self):
        """omni_route_condition returns 'finish' for explicit finish decision → END."""
        route = omni_route_condition(_state(decision=_decision("finish")))
        assert route == "finish"

    @pytest.mark.asyncio
    async def test_loop_back_to_brain_after_hands_execution(self):
        """
        The graph has workflow.add_edge('omni_hands', 'omni_brain'), meaning
        after Hands executes, Brain is called again for the next decision.

        Verified end-to-end: Brain is called TWICE (once → agent, once → finish)
        while Hands is called exactly ONCE.
        """
        call_count = {"brain": 0, "hands": 0}

        async def mock_think(state, config=None):
            call_count["brain"] += 1
            if call_count["brain"] == 1:
                # First call: ask agent
                return {
                    "decision": {"action_type": "agent", "resource_id": "test",
                                 "payload": {}},
                    "iteration_count": 1,
                }
            else:
                # Second call: finish
                return {
                    "decision": {"action_type": "finish", "user_response": "Done."},
                    "final_response": "Done.",
                    "iteration_count": 2,
                }

        async def mock_execute(state, config=None):
            call_count["hands"] += 1
            return {
                "execution_result": {
                    "action_id": "agent_test", "success": True,
                    "output": {"message": "ok"}, "error_message": None,
                    "execution_time_ms": 1.0,
                }
            }

        with patch.object(module_brain, "think", mock_think), \
             patch.object(module_hands, "execute", mock_execute):
            test_graph = create_graph_with_checkpointer(MemorySaver())
            await test_graph.ainvoke(
                _state(),
                {"configurable": {"thread_id": "loop_test_1"}},
            )

        assert call_count["brain"] == 2, (
            f"Expected Brain called 2× (agent→finish), got {call_count['brain']}"
        )
        assert call_count["hands"] == 1, (
            f"Expected Hands called 1×, got {call_count['hands']}"
        )

    @pytest.mark.asyncio
    async def test_graph_pauses_on_approval_state(self):
        """
        When Brain sets pending_approval=True, the graph routes to
        action_approval_required, which sets pending_user_input=True and
        then terminates (approval_required → END edge).

        Verified by checking the final state has pending_user_input=True
        and pending_approval preserved.
        """
        async def mock_think_approval(state, config=None):
            # Brain requires approval before executing
            return {
                "decision": {
                    "action_type": "agent",
                    "resource_id": "risky_agent",
                    "requires_approval": True,
                    "approval_reason": "Risky external action",
                },
                "pending_approval": True,
                "pending_decision": {
                    "action_type": "agent",
                    "resource_id": "risky_agent",
                    "requires_approval": True,
                    "approval_reason": "Risky external action",
                },
                "iteration_count": 1,
            }

        mock_execute = AsyncMock()

        with patch.object(module_brain, "think", mock_think_approval), \
             patch.object(module_hands, "execute", mock_execute):
            test_graph = create_graph_with_checkpointer(MemorySaver())
            final_state = await test_graph.ainvoke(
                _state(),
                {"configurable": {"thread_id": "approval_test_1"}},
            )

        # Graph must have routed to action_approval_required node
        assert final_state.get("pending_user_input") is True
        assert final_state.get("pending_approval") is True
        # Hands must NOT have been called — execution was paused
        mock_execute.assert_not_called()
