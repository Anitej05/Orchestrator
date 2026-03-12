"""
Unit tests for backend/orchestrator/hands.py

Tests the Hands dispatcher's deterministic logic without hitting real agents,
file systems, or network endpoints. All external calls are mocked.

Coverage:
  1.  TestPythonActionBlockedDangerousImports (3 tests)
        a. SyntaxError caught pre-sandbox (security gate)
        b. Empty code rejected without touching sandbox
        c. Dangerous imports NOT blocked at Hands layer (documents known gap)

  2.  TestTimeoutEnforcedPerActionType (3 tests)
        a. python action timeout → failed ActionResult with 'timed out' message
        b. terminal action timeout → failed ActionResult
        c. agent action uses per-agent timeout (_get_agent_timeout + 30s), not 120s

  3.  TestParallelActionUsesAsyncioGather (2 tests)
        a. asyncio.gather called for parallel actions (fanout verified)
        b. empty parallel_actions list → failed ActionResult

  4.  TestTaskEventCallbacks (3 tests)
        a. task_started fired before agent execution begins
        b. task_completed fired after successful agent execution
        c. task_failed fired after failed agent execution

  5.  TestLastAgentResultInjectedIntoState (2 tests)
        a. successful agent call → last_agent_result set with success=True
        b. failed agent call → last_agent_result cleared to None

  6.  TestAgentActionPostsToCorrectPort (2 tests)
        a. resource_id passed unchanged to agent_manager.execute()
        b. instruction forwarded as 'prompt' in task dict

  7.  TestPythonActionRunsInSandbox (2 tests)
        a. code_sandbox.execute_code called with user's code
        b. code wrapped with os.chdir(workspace_path) header

  8.  TestFinishActionExtractsFinalResponse (3 tests)
        a. finish → execution_result.output['message'] == user_response
        b. finish with code block → canvas_display auto-detected via CanvasService
        c. skip → execution_result.output['skipped'] == True

Patching notes (CRITICAL — lazy imports require patching at source module):
  - get_shared_workspace_manager  → backend.orchestrator.shared_workspace
  - get_canvas_registry           → backend.services.canvas_registry
  - get_artifact_store            → backend.orchestrator.artifact_store
  - get_workspace_manager         → backend.orchestrator.hands  (top-level import, OK)
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

from backend.orchestrator.hands import Hands
from backend.orchestrator.schemas import ActionResult, TaskStatus


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_workspace_mock(workspace_path: str = "/tmp/test_ws") -> MagicMock:
    wm = MagicMock()
    wm.get_workspace_path.return_value = Path(workspace_path)
    wm.scan_for_new_files.return_value = []
    wm.list_files.return_value = []
    return wm


def _make_shared_workspace_mock() -> MagicMock:
    swm = MagicMock()
    swm.get_workspace_path.return_value = Path("/tmp/shared_ws")
    swm.list_files.return_value = []
    return swm


def _make_canvas_registry_mock() -> MagicMock:
    reg = MagicMock()
    reg.get_registry_state.return_value = MagicMock(model_dump=lambda: {})
    reg.get_active_id.return_value = None
    reg.get_backward_compat_fields.return_value = {}
    return reg


def _base_state(**overrides) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "original_prompt": "Test prompt",
        "todo_list": [
            {"task_id": "task_1", "description": "Test task", "status": "pending"}
        ],
        "memory": {},
        "insights": {},
        "action_history": [],
        "iteration_count": 1,
        "failure_count": 0,
        "execution_result": None,
        "last_agent_result": None,
        "current_task_id": "task_1",
        "uploaded_files": [],
        "user_id": "test_user",
        "decision": {},
    }
    state.update(overrides)
    return state


def _make_config(
    task_event_callback=None, thread_id: str = "test_thread", user_id: str = "test_user"
) -> Dict:
    cfg: Dict[str, Any] = {
        "configurable": {
            "thread_id": thread_id,
            "owner": {"user_id": user_id},
        }
    }
    if task_event_callback is not None:
        cfg["configurable"]["task_event_callback"] = task_event_callback
    return cfg


def _make_agent_manager_mock(result: dict = None, raise_exc: Exception = None) -> MagicMock:
    mgr = MagicMock()
    mgr._initialized = True
    if raise_exc:
        mgr.execute = AsyncMock(side_effect=raise_exc)
    else:
        mgr.execute = AsyncMock(return_value=result or {"status": "ok", "message": "done"})
    return mgr


def _enter_update_state_patches(stack: ExitStack):
    """
    Enter the three patches that _update_state_with_result needs.

    All three are LAZY imports (inside the method body), so they MUST be patched
    at their source module, not at 'backend.orchestrator.hands'.
    """
    stack.enter_context(patch(
        "backend.orchestrator.shared_workspace.get_shared_workspace_manager",
        return_value=_make_shared_workspace_mock(),
    ))
    # artifact capture is fire-and-forget wrapped in try/except; raising silently is fine
    stack.enter_context(patch(
        "backend.orchestrator.artifact_store.get_artifact_store",
        side_effect=Exception("test: skip artifact capture"),
    ))
    # canvas registry only called when a canvas is found in the result
    stack.enter_context(patch(
        "backend.services.canvas_registry.get_canvas_registry",
        return_value=_make_canvas_registry_mock(),
    ))


def _enter_agent_patches(stack: ExitStack, agent_manager_mock: MagicMock,
                          agent_name: str = "spreadsheet_agent"):
    """Enter patches required by the agent execution path."""
    stack.enter_context(patch(
        "backend.orchestrator.hands.agent_registry",
        **{"find_agent.return_value": {"name": agent_name}},
    ))
    stack.enter_context(patch(
        "backend.services.agent_manager.get_agent_manager",
        return_value=agent_manager_mock,
    ))
    stack.enter_context(patch(
        "backend.services.agent_manager._get_agent_timeout",
        return_value=120.0,
    ))


def _enter_common_patches(stack: ExitStack, workspace_path: str = "/tmp/test_ws"):
    """Enter patches shared by all full execute() tests."""
    stack.enter_context(patch(
        "backend.orchestrator.hands.get_workspace_manager",
        return_value=_make_workspace_mock(workspace_path),
    ))
    stack.enter_context(patch(
        "backend.orchestrator.hands.hooks.on_task_complete",
        new_callable=AsyncMock,
        return_value={},
    ))
    stack.enter_context(patch("backend.orchestrator.hands.telemetry_service"))
    _enter_update_state_patches(stack)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Security: syntax / empty-code / dangerous-import blocking
# ─────────────────────────────────────────────────────────────────────────────

class TestPythonActionBlockedDangerousImports:
    """
    Security-critical gate in hands._execute_python.

    compile() pre-validates syntax BEFORE code reaches code_sandbox.execute_code.
    This is the first (and currently only) safety check at the Hands layer.
    """

    @pytest.mark.asyncio
    async def test_syntax_error_rejected_before_sandbox(self):
        """SyntaxError in LLM-generated code → failed ActionResult; sandbox never called."""
        hands = Hands()
        mock_sandbox = MagicMock()
        mock_sandbox.execute_code = MagicMock()

        bad_code = "def oops(\n    x = 1\nprint(x)"  # unclosed function def

        with patch("backend.orchestrator.hands.code_sandbox", mock_sandbox), \
             patch("backend.orchestrator.hands.get_workspace_manager",
                   return_value=_make_workspace_mock()):
            result = await hands._execute_python(
                {"code": bad_code}, start_time=0.0, thread_id="t1", state={}
            )

        assert result.success is False
        assert "syntax" in (result.error_message or "").lower()
        mock_sandbox.execute_code.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_code_rejected_without_touching_sandbox(self):
        """Empty code payload → immediate failure; sandbox not invoked."""
        hands = Hands()
        mock_sandbox = MagicMock()
        mock_sandbox.execute_code = MagicMock()

        with patch("backend.orchestrator.hands.code_sandbox", mock_sandbox), \
             patch("backend.orchestrator.hands.get_workspace_manager",
                   return_value=_make_workspace_mock()):
            result = await hands._execute_python({"code": ""}, start_time=0.0)

        assert result.success is False
        assert "no code" in (result.error_message or "").lower()
        mock_sandbox.execute_code.assert_not_called()

    @pytest.mark.asyncio
    async def test_dangerous_import_subprocess_not_blocked_at_hands_layer(self):
        """
        SECURITY GAP DOCUMENTATION TEST.

        Dangerous imports (subprocess, ctypes) are NOT blocked at the Hands layer.
        Code is forwarded to code_sandbox.execute_code as-is.

        If hands-level import blocking is ever added, update this test to assert
        success=False and that the sandbox is NOT called.
        """
        hands = Hands()
        mock_sandbox = MagicMock()
        mock_sandbox.execute_code.return_value = {
            "success": True, "stdout": "hello", "result": None, "error": None
        }

        dangerous_code = (
            "import subprocess\n"
            "r = subprocess.run(['echo', 'hello'], capture_output=True, text=True)\n"
            "print(r.stdout)"
        )

        with patch("backend.orchestrator.hands.code_sandbox", mock_sandbox), \
             patch("backend.orchestrator.hands.get_workspace_manager",
                   return_value=_make_workspace_mock()):
            await hands._execute_python(
                {"code": dangerous_code}, start_time=0.0, thread_id="t1", state={}
            )

        mock_sandbox.execute_code.assert_called_once()
        called_code = mock_sandbox.execute_code.call_args[0][0]
        assert "subprocess" in called_code  # code forwarded unchanged


# ─────────────────────────────────────────────────────────────────────────────
# 2. Timeout enforcement per action type
# ─────────────────────────────────────────────────────────────────────────────

class TestTimeoutEnforcedPerActionType:
    """
    asyncio.wait_for wraps each direct-execution action.
    On TimeoutError, Hands returns a failed ActionResult with a meaningful message.

    Strategy: patch the private _execute_* method to raise asyncio.TimeoutError,
    which propagates through wait_for and is caught by the outer except block.
    """

    @pytest.mark.asyncio
    async def test_timeout_for_python_action(self):
        """asyncio.TimeoutError from python action → success=False, 'timed out' in message."""
        hands = Hands()
        state = _base_state(decision={
            "action_type": "python",
            "payload": {"code": "import time; time.sleep(9999)"},
        })

        with ExitStack() as stack:
            stack.enter_context(patch.object(
                Hands, "_execute_python", AsyncMock(side_effect=asyncio.TimeoutError)
            ))
            stack.enter_context(patch(
                "backend.orchestrator.hands.agent_registry"
            ))
            _enter_common_patches(stack)
            updates = await hands.execute(state, _make_config())

        result_dict = updates.get("execution_result", {})
        assert result_dict.get("success") is False
        assert "timed out" in (result_dict.get("error_message") or "").lower()

    @pytest.mark.asyncio
    async def test_timeout_for_terminal_action(self):
        """asyncio.TimeoutError from terminal action → success=False ActionResult."""
        hands = Hands()
        state = _base_state(decision={
            "action_type": "terminal",
            "payload": {"command": "sleep 9999"},
        })

        with ExitStack() as stack:
            stack.enter_context(patch.object(
                Hands, "_execute_terminal", AsyncMock(side_effect=asyncio.TimeoutError)
            ))
            stack.enter_context(patch("backend.orchestrator.hands.agent_registry"))
            _enter_common_patches(stack)
            updates = await hands.execute(state, _make_config())

        result_dict = updates.get("execution_result", {})
        assert result_dict.get("success") is False
        assert "timed out" in (result_dict.get("error_message") or "").lower()

    @pytest.mark.asyncio
    async def test_agent_uses_per_agent_timeout_not_generic_map(self):
        """
        Agent execution uses _get_agent_timeout(agent_id) + 30s buffer, not 120s.
        Verified by spying on asyncio.wait_for's timeout argument.
        """
        hands = Hands()
        captured_timeouts = []
        _real_wait_for = asyncio.wait_for

        async def spy_wait_for(coro, timeout=None):
            captured_timeouts.append(timeout)
            return await _real_wait_for(coro, timeout=60.0)

        agent_mgr = _make_agent_manager_mock(result={"status": "ok", "message": "done"})
        state = _base_state(decision={
            "action_type": "agent",
            "resource_id": "browser_automation_agent",
            "payload": {"instruction": "browse the web"},
        })

        with ExitStack() as stack:
            stack.enter_context(patch("asyncio.wait_for", side_effect=spy_wait_for))
            # Patch _get_agent_timeout ONCE here — do NOT call _enter_agent_patches which
            # would add a second patch for 120.0 that would shadow this 900.0 patch.
            stack.enter_context(patch(
                "backend.services.agent_manager._get_agent_timeout", return_value=900.0
            ))
            stack.enter_context(patch(
                "backend.orchestrator.hands.agent_registry",
                **{"find_agent.return_value": {"name": "browser_automation_agent"}},
            ))
            stack.enter_context(patch(
                "backend.services.agent_manager.get_agent_manager", return_value=agent_mgr
            ))
            _enter_common_patches(stack)
            await hands.execute(state, _make_config())

        assert len(captured_timeouts) >= 1
        assert captured_timeouts[0] == 930.0  # 900 + 30s buffer


# ─────────────────────────────────────────────────────────────────────────────
# 3. Parallel action uses asyncio.gather
# ─────────────────────────────────────────────────────────────────────────────

class TestParallelActionUsesAsyncioGather:

    @pytest.mark.asyncio
    async def test_parallel_uses_gather_for_multiple_actions(self):
        """asyncio.gather is invoked when two parallel agent actions are dispatched."""
        hands = Hands()
        gather_arities = []
        _real_gather = asyncio.gather

        async def spy_gather(*coros, **kwargs):
            gather_arities.append(len(coros))
            return await _real_gather(*coros, **kwargs)

        agent_mgr = _make_agent_manager_mock(result={"status": "ok", "text": "result"})
        state = _base_state(decision={
            "action_type": "parallel",
            "parallel_actions": [
                {"action_type": "agent", "resource_id": "spreadsheet_agent",
                 "payload": {"instruction": "do A"}},
                {"action_type": "agent", "resource_id": "document_agent",
                 "payload": {"instruction": "do B"}},
            ],
        })

        with ExitStack() as stack:
            stack.enter_context(patch("asyncio.gather", side_effect=spy_gather))
            _enter_common_patches(stack)
            _enter_agent_patches(stack, agent_mgr)
            updates = await hands.execute(state, _make_config())

        # gather called exactly once with 2 task coroutines
        assert len(gather_arities) == 1
        assert gather_arities[0] == 2

    @pytest.mark.asyncio
    async def test_empty_parallel_actions_returns_failure(self):
        """parallel action with no sub-actions → failed ActionResult."""
        hands = Hands()
        state = _base_state(decision={"action_type": "parallel", "parallel_actions": []})

        with ExitStack() as stack:
            stack.enter_context(patch("backend.orchestrator.hands.agent_registry"))
            _enter_common_patches(stack)
            updates = await hands.execute(state, _make_config())

        result_dict = updates.get("execution_result", {})
        assert result_dict.get("success") is False
        assert "no parallel actions" in (result_dict.get("error_message") or "").lower()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Task event callbacks
# ─────────────────────────────────────────────────────────────────────────────

class TestTaskEventCallbacks:

    def _agent_state(self):
        return _base_state(decision={
            "action_type": "agent",
            "resource_id": "spreadsheet_agent",
            "payload": {"instruction": "analyze data"},
        })

    @pytest.mark.asyncio
    async def test_task_started_fired_before_agent_execution(self):
        """task_started event is emitted BEFORE agent_manager.execute() is awaited."""
        hands = Hands()
        events = []
        snapshot_at_exec = []

        async def capture_event(evt):
            events.append(evt)

        async def tracked_execute(agent_id, task, progress_callback=None):
            snapshot_at_exec.append(list(events))   # snapshot the log before completion
            return {"status": "ok", "message": "done"}

        agent_mgr = MagicMock()
        agent_mgr._initialized = True
        agent_mgr.execute = tracked_execute

        with ExitStack() as stack:
            _enter_common_patches(stack)
            _enter_agent_patches(stack, agent_mgr)
            await hands.execute(self._agent_state(), _make_config(task_event_callback=capture_event))

        assert len(snapshot_at_exec) == 1
        started = [e for e in snapshot_at_exec[0] if e.get("event_type") == "task_started"]
        assert len(started) == 1

    @pytest.mark.asyncio
    async def test_task_completed_fired_after_agent_success(self):
        """task_completed event emitted when agent execution succeeds."""
        hands = Hands()
        events = []

        async def capture_event(evt):
            events.append(evt)

        agent_mgr = _make_agent_manager_mock(
            result={"status": "ok", "text_response": "Analysis complete."}
        )

        with ExitStack() as stack:
            stack.enter_context(patch(
                "backend.orchestrator.hands.hooks.on_task_complete",
                new_callable=AsyncMock,
                return_value={"text_response": "Analysis complete."},
            ))
            stack.enter_context(patch(
                "backend.orchestrator.hands.get_workspace_manager",
                return_value=_make_workspace_mock(),
            ))
            stack.enter_context(patch("backend.orchestrator.hands.telemetry_service"))
            _enter_update_state_patches(stack)
            _enter_agent_patches(stack, agent_mgr)
            await hands.execute(self._agent_state(), _make_config(task_event_callback=capture_event))

        completed = [e for e in events if e.get("event_type") == "task_completed"]
        assert len(completed) == 1
        assert completed[0]["task_name"] == "task_1"

    @pytest.mark.asyncio
    async def test_task_failed_fired_after_agent_failure(self):
        """task_failed event emitted when agent returns success=False."""
        hands = Hands()
        events = []

        async def capture_event(evt):
            events.append(evt)

        agent_mgr = _make_agent_manager_mock(
            result={"status": "error", "success": False, "error": "Agent crashed"}
        )

        with ExitStack() as stack:
            _enter_common_patches(stack)
            _enter_agent_patches(stack, agent_mgr)
            await hands.execute(self._agent_state(), _make_config(task_event_callback=capture_event))

        failed = [e for e in events if e.get("event_type") == "task_failed"]
        assert len(failed) == 1
        assert failed[0]["task_name"] == "task_1"


# ─────────────────────────────────────────────────────────────────────────────
# 5. last_agent_result injected into state
# ─────────────────────────────────────────────────────────────────────────────

class TestLastAgentResultInjectedIntoState:

    @pytest.mark.asyncio
    async def test_last_agent_result_set_on_success(self):
        """After a successful agent call, last_agent_result has success=True and result text."""
        hands = Hands()
        agent_result = {"status": "ok", "text_response": "The answer is 42."}
        agent_mgr = _make_agent_manager_mock(result=agent_result)

        state = _base_state(decision={
            "action_type": "agent",
            "resource_id": "universal_agent",
            "payload": {"instruction": "answer question"},
        })

        with ExitStack() as stack:
            stack.enter_context(patch(
                "backend.orchestrator.hands.hooks.on_task_complete",
                new_callable=AsyncMock,
                return_value=agent_result,
            ))
            stack.enter_context(patch(
                "backend.orchestrator.hands.get_workspace_manager",
                return_value=_make_workspace_mock(),
            ))
            stack.enter_context(patch("backend.orchestrator.hands.telemetry_service"))
            _enter_update_state_patches(stack)
            _enter_agent_patches(stack, agent_mgr, "universal_agent")
            updates = await hands.execute(state, _make_config())

        lar = updates.get("last_agent_result")
        assert lar is not None, "last_agent_result must be set after agent success"
        assert lar["success"] is True
        assert lar["agent"] == "universal_agent"
        assert "42" in lar["result"]

    @pytest.mark.asyncio
    async def test_last_agent_result_cleared_on_failure(self):
        """After a failed agent call, last_agent_result is set to None to avoid stale reads."""
        hands = Hands()
        agent_mgr = _make_agent_manager_mock(
            result={"status": "error", "success": False, "error": "failed"}
        )

        state = _base_state(
            last_agent_result={"agent": "old", "success": True, "result": "stale"},
            decision={
                "action_type": "agent",
                "resource_id": "universal_agent",
                "payload": {"instruction": "do something"},
            },
        )

        with ExitStack() as stack:
            _enter_common_patches(stack)
            _enter_agent_patches(stack, agent_mgr, "universal_agent")
            updates = await hands.execute(state, _make_config())

        assert updates.get("last_agent_result") is None


class TestNeedsInputHandling:

    @pytest.mark.asyncio
    async def test_nested_needs_input_pauses_workflow_and_keeps_task_in_progress(self):
        """Nested agent needs_input payloads must pause the graph instead of completing the task."""
        hands = Hands()
        state = _base_state(
            decision={
                "action_type": "agent",
                "resource_id": "gmail_agent",
                "payload": {"prompt": "Read unread emails"},
            }
        )

        raw_agent_output = {
            "status": "needs_input",
            "question": "Please confirm the connected Gmail account.",
        }

        result = ActionResult(
            action_id="agent_gmail_agent",
            success=True,
            output={"result": raw_agent_output, "status": "completed"},
            execution_time_ms=12.0,
        )
        result._raw_output = raw_agent_output

        with ExitStack() as stack:
            _enter_common_patches(stack)
            updates = hands._update_state_with_result(state, result, _make_config())

        assert updates["pending_user_input"] is True
        assert updates["question_for_user"] == "Please confirm the connected Gmail account."
        assert updates["last_agent_result"] is None
        assert updates["todo_list"][0]["status"] == TaskStatus.IN_PROGRESS


# ─────────────────────────────────────────────────────────────────────────────
# 6. Agent action posts to correct agent_id
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentActionPostsToCorrectPort:

    @pytest.mark.asyncio
    async def test_correct_agent_id_passed_to_manager(self):
        """agent_manager.execute is called with exactly the resource_id from the decision."""
        hands = Hands()
        captured_calls = []

        async def capture_execute(agent_id, task, progress_callback=None):
            captured_calls.append({"agent_id": agent_id, "task": task})
            return {"status": "ok", "message": "done"}

        agent_mgr = MagicMock()
        agent_mgr._initialized = True
        agent_mgr.execute = capture_execute

        state = _base_state(decision={
            "action_type": "agent",
            "resource_id": "spreadsheet_agent",
            "payload": {"instruction": "compute average", "thread_id": "t1"},
        })

        with ExitStack() as stack:
            _enter_common_patches(stack)
            _enter_agent_patches(stack, agent_mgr)
            await hands.execute(state, _make_config())

        assert len(captured_calls) == 1
        assert captured_calls[0]["agent_id"] == "spreadsheet_agent"

    @pytest.mark.asyncio
    async def test_instruction_forwarded_from_payload(self):
        """The 'instruction' from the decision payload becomes 'prompt' in the task dict."""
        hands = Hands()
        captured_tasks = []

        async def capture_execute(agent_id, task, progress_callback=None):
            captured_tasks.append(task)
            return {"status": "ok", "message": "done"}

        agent_mgr = MagicMock()
        agent_mgr._initialized = True
        agent_mgr.execute = capture_execute

        instruction_text = "Compute the mean for column B across all rows"
        state = _base_state(decision={
            "action_type": "agent",
            "resource_id": "spreadsheet_agent",
            "payload": {"instruction": instruction_text},
        })

        with ExitStack() as stack:
            _enter_common_patches(stack)
            _enter_agent_patches(stack, agent_mgr)
            await hands.execute(state, _make_config())

        assert len(captured_tasks) == 1
        assert captured_tasks[0]["prompt"] == instruction_text


# ─────────────────────────────────────────────────────────────────────────────
# 7. Python action runs in sandbox
# ─────────────────────────────────────────────────────────────────────────────

class TestPythonActionRunsInSandbox:

    @pytest.mark.asyncio
    async def test_execute_code_called_with_user_code(self):
        """code_sandbox.execute_code called once; modified code contains user's code."""
        hands = Hands()
        mock_sandbox = MagicMock()
        mock_sandbox.execute_code.return_value = {
            "success": True, "stdout": "42\n", "result": 42, "error": None
        }

        user_code = "result = 6 * 7\nprint(result)"

        with patch("backend.orchestrator.hands.code_sandbox", mock_sandbox), \
             patch("backend.orchestrator.hands.get_workspace_manager",
                   return_value=_make_workspace_mock()):
            result = await hands._execute_python(
                {"code": user_code}, start_time=0.0, thread_id="t1", state={}
            )

        mock_sandbox.execute_code.assert_called_once()
        called_code = mock_sandbox.execute_code.call_args[0][0]
        assert "result = 6 * 7" in called_code
        assert result.success is True

    @pytest.mark.asyncio
    async def test_code_wrapped_with_workspace_chdir(self):
        """Code sent to sandbox is prepended with os.chdir(workspace_path)."""
        hands = Hands()
        mock_sandbox = MagicMock()
        mock_sandbox.execute_code.return_value = {
            "success": True, "stdout": "", "result": None, "error": None
        }

        workspace_path = "/tmp/my_workspace"

        with patch("backend.orchestrator.hands.code_sandbox", mock_sandbox), \
             patch("backend.orchestrator.hands.get_workspace_manager",
                   return_value=_make_workspace_mock(workspace_path)):
            await hands._execute_python(
                {"code": "x = 1"}, start_time=0.0, thread_id="t1", state={}
            )

        called_code = mock_sandbox.execute_code.call_args[0][0]
        assert "os.chdir" in called_code
        # Path() normalises slashes on the current OS; compare against normalised form
        assert str(Path(workspace_path)) in called_code


# ─────────────────────────────────────────────────────────────────────────────
# 8. Finish / skip action extracts final response
# ─────────────────────────────────────────────────────────────────────────────

class TestFinishActionExtractsFinalResponse:

    @pytest.mark.asyncio
    async def test_finish_sets_output_message(self):
        """finish → execution_result.output['message'] == user_response from decision."""
        hands = Hands()
        user_response = "The spreadsheet contains 1,240 rows with a total of $84,000."
        state = _base_state(decision={
            "action_type": "finish",
            "user_response": user_response,
        })

        with ExitStack() as stack:
            stack.enter_context(patch("backend.orchestrator.hands.agent_registry"))
            _enter_common_patches(stack)
            updates = await hands.execute(state, _make_config())

        result_dict = updates.get("execution_result", {})
        assert result_dict.get("success") is True
        assert result_dict["output"]["message"] == user_response

    @pytest.mark.asyncio
    async def test_finish_auto_detects_canvas_for_code_block(self):
        """
        finish with a 5+ line code block in user_response triggers
        _auto_detect_canvas_from_text and attaches canvas_display to output.
        """
        hands = Hands()

        code_response = (
            "Here is the solution:\n"
            "```python\n"
            "def solve():\n"
            "    data = load_data()\n"
            "    result = process(data)\n"
            "    visualize(result)\n"
            "    return result\n"
            "```"
        )

        mock_canvas = {"canvas_type": "code_viewer", "canvas_content": "...", "canvas_data": {}}
        mock_canvas_obj = MagicMock()
        mock_canvas_obj.model_dump.return_value = mock_canvas

        state = _base_state(decision={
            "action_type": "finish",
            "user_response": code_response,
        })

        with ExitStack() as stack:
            stack.enter_context(patch("backend.orchestrator.hands.agent_registry"))
            mock_canvas_svc = stack.enter_context(
                patch("backend.orchestrator.hands.CanvasService")
            )
            _enter_common_patches(stack)
            mock_canvas_svc.build_from_template.return_value = mock_canvas_obj
            updates = await hands.execute(state, _make_config())

        result_dict = updates.get("execution_result", {})
        assert result_dict.get("success") is True
        assert result_dict["output"].get("canvas_display") == mock_canvas

    @pytest.mark.asyncio
    async def test_skip_action_returns_skipped_true(self):
        """skip action → execution_result.output['skipped'] == True; nothing executed."""
        hands = Hands()
        state = _base_state(decision={"action_type": "skip"})

        with ExitStack() as stack:
            stack.enter_context(patch("backend.orchestrator.hands.agent_registry"))
            _enter_common_patches(stack)
            updates = await hands.execute(state, _make_config())

        result_dict = updates.get("execution_result", {})
        assert result_dict.get("success") is True
        assert result_dict["output"].get("skipped") is True
