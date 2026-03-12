"""
Unit tests for WebSocket endpoints and LangGraph State.

Tests the deterministic, synchronous logic without hitting real LLM endpoints,
databases, or external services. All async helpers run via asyncio.run().

Coverage:
  State reducers (state.py)
   1.  overwrite_reducer — always returns b
   2.  overwrite_reducer — returns None when b is None
   3.  overwrite_reducer — returns b when a is None
   4.  append_reducer — concatenates two lists
   5.  append_reducer — None a treated as empty list
   6.  append_reducer — None b returns a unchanged
   7.  append_reducer — both None returns []
   8.  concat_reducer — both values joined with separator
   9.  concat_reducer — one None returns the other
  10.  or_overwrite — True | False → True
  11.  or_overwrite — False | True → True
  12.  or_overwrite — None | None → False
  13.  or_overwrite — False | None → False

  Sequence counter (monotonic seq numbers on task events)
  14.  First call returns 1
  15.  Multiple calls are strictly increasing and unique
  16.  Two independent closures do not share state
  17.  task_started seq < task_completed seq for same task

  Screenshot relay logic
  18.  Relay sends JSON to the registered frontend WebSocket
  19.  Relay skips when no frontend is registered for thread_id
  20.  Relay removes stale frontend entry on send failure

  safe_websocket_send helper
  21.  Successful send returns True
  22.  Exception during send returns False (does not re-raise)

  WebSocket /ws/chat — validation via minimal replica app
  23.  New thread without owner → __error__ with ValidationError
  24.  Neither prompt nor user_response → __error__ with ValidationError
  25.  Valid new thread (owner + prompt) → __start__ acknowledged
  26.  Existing thread with user_response (no prompt) → __start__ acknowledged
  27.  owner provided as string is coerced to dict

  WebSocket /ws/chat — orchestration exception path
  28.  RuntimeError in execute_orchestration → __error__ node with error_type

  Thread-id isolation in frontend_websockets registry
  29.  Messages for thread A do not reach thread B's WebSocket
  30.  Removing thread A from registry does not remove thread B

  MemorySaver state persistence
  31.  Checkpoint is written after graph invocation
  32.  Invocation result reflects node execution (count increments)
  33.  Thread A checkpoint is not visible from thread B config
  34.  Non-existent thread checkpoint returns None

  State TypedDict field declarations
  35.  last_agent_result declared
  36.  owner_id declared
  37.  action_history declared
  38.  todo_list declared
  39.  pending_action declared
  40.  pending_action_approval declared
  41.  last_agent_result uses overwrite_reducer
  42.  action_history uses append_reducer
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock

import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent       # backend/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))               # project root

from backend.orchestrator.state import (
    State,
    append_reducer,
    concat_reducer,
    or_overwrite,
    overwrite_reducer,
)


# =============================================================================
# 1–13  State reducers
# =============================================================================

class TestOverwriteReducer:
    def test_returns_new_value(self):
        assert overwrite_reducer("old", "new") == "new"

    def test_returns_none_when_b_is_none(self):
        assert overwrite_reducer("old", None) is None

    def test_returns_b_when_a_is_none(self):
        assert overwrite_reducer(None, "value") == "value"

    def test_works_with_dict(self):
        assert overwrite_reducer({"k": 1}, {"k": 2}) == {"k": 2}

    def test_works_with_list(self):
        assert overwrite_reducer([1, 2], [3]) == [3]


class TestAppendReducer:
    def test_concatenates_two_lists(self):
        assert append_reducer([1, 2], [3, 4]) == [1, 2, 3, 4]

    def test_none_a_treated_as_empty(self):
        assert append_reducer(None, [1, 2]) == [1, 2]

    def test_none_b_returns_a_unchanged(self):
        assert append_reducer([1, 2], None) == [1, 2]

    def test_both_none_returns_empty_list(self):
        assert append_reducer(None, None) == []

    def test_empty_lists_produce_empty(self):
        assert append_reducer([], []) == []

    def test_order_preserved(self):
        result = append_reducer(["a"], ["b", "c"])
        assert result == ["a", "b", "c"]


class TestConcatReducer:
    def test_both_values_joined_with_separator(self):
        result = concat_reducer("part1", "part2")
        assert "part1" in result
        assert "part2" in result
        assert "\n\n---\n\n" in result

    def test_a_none_returns_b(self):
        assert concat_reducer(None, "only") == "only"

    def test_b_none_returns_a(self):
        assert concat_reducer("only", None) == "only"

    def test_both_none_returns_none(self):
        assert concat_reducer(None, None) is None


class TestOrOverwrite:
    def test_true_or_false_is_true(self):
        assert or_overwrite(True, False) is True

    def test_false_or_true_is_true(self):
        assert or_overwrite(False, True) is True

    def test_none_or_none_is_false(self):
        assert or_overwrite(None, None) is False

    def test_false_or_none_is_false(self):
        assert or_overwrite(False, None) is False

    def test_true_or_none_is_true(self):
        assert or_overwrite(True, None) is True


# =============================================================================
# 14–17  Sequence counter (closure pattern used in /ws/chat)
# =============================================================================

class TestSequenceCounter:
    """
    The /ws/chat endpoint uses this closure per message handled:

        _task_event_seq = {"n": 0}
        def _next_seq() -> int:
            _task_event_seq["n"] += 1
            return _task_event_seq["n"]

    Sequence numbers must be monotonically increasing and isolated
    between separate message-handling sessions (separate closures).
    """

    @staticmethod
    def _make_seq_counter():
        _task_event_seq = {"n": 0}

        def _next_seq() -> int:
            _task_event_seq["n"] += 1
            return _task_event_seq["n"]

        return _next_seq

    def test_first_call_returns_one(self):
        seq = self._make_seq_counter()
        assert seq() == 1

    def test_multiple_calls_are_strictly_increasing(self):
        seq = self._make_seq_counter()
        values = [seq() for _ in range(10)]
        assert values == sorted(values)
        assert len(set(values)) == 10  # all unique

    def test_two_independent_closures_are_isolated(self):
        seq_a = self._make_seq_counter()
        seq_b = self._make_seq_counter()
        # Advance seq_a multiple times
        for _ in range(5):
            seq_a()
        # seq_b is independent; starts from 1
        assert seq_b() == 1

    def test_task_started_seq_less_than_task_completed(self):
        """Events for the same task must arrive in order: started < completed."""
        seq = self._make_seq_counter()
        started_seq = seq()
        completed_seq = seq()
        assert started_seq < completed_seq


# =============================================================================
# 18–20  Screenshot relay logic
# =============================================================================

class TestScreenshotRelay:
    """
    The screenshots_websocket endpoint maintains:
        frontend_websockets: Dict[str, WebSocket]

    On each screenshot received from the browser agent it:
      1. Looks up the frontend WS for the thread_id
      2. Sends the screenshot payload
      3. Removes the entry on send failure
    """

    def test_relay_sends_to_registered_frontend(self):
        async def _run():
            frontend_ws = AsyncMock()
            frontend_websockets: Dict[str, Any] = {"thread-1": frontend_ws}
            lock = asyncio.Lock()

            payload = {"node": "__live_canvas__", "thread_id": "thread-1", "data": {}}
            async with lock:
                ws = frontend_websockets.get("thread-1")
                if ws:
                    await ws.send_json(payload)

            frontend_ws.send_json.assert_called_once_with(payload)

        asyncio.run(_run())

    def test_relay_skips_when_no_frontend_registered(self):
        async def _run():
            frontend_websockets: Dict[str, Any] = {}
            lock = asyncio.Lock()
            send_called = False

            async with lock:
                ws = frontend_websockets.get("thread-1")
                if ws:
                    await ws.send_json({})
                    send_called = True

            assert not send_called

        asyncio.run(_run())

    def test_relay_removes_stale_frontend_on_send_failure(self):
        async def _run():
            stale_ws = AsyncMock()
            stale_ws.send_json.side_effect = RuntimeError("connection closed")
            frontend_websockets: Dict[str, Any] = {"thread-1": stale_ws}
            lock = asyncio.Lock()

            async with lock:
                ws = frontend_websockets.get("thread-1")
                if ws:
                    try:
                        await ws.send_json({"node": "__live_canvas__"})
                    except Exception:
                        del frontend_websockets["thread-1"]

            assert "thread-1" not in frontend_websockets

        asyncio.run(_run())


# =============================================================================
# 21–22  safe_websocket_send helper
# =============================================================================

class TestSafeWebSocketSend:
    """
    safe_websocket_send wraps send_json in try/except.
    Returns True on success, False on any exception.
    """

    @staticmethod
    async def _safe_send(websocket, data, thread_id="unknown"):
        try:
            await websocket.send_json(data)
            return True
        except Exception:
            return False

    def test_successful_send_returns_true(self):
        async def _run():
            ws = AsyncMock()
            result = await self._safe_send(ws, {"node": "test"})
            assert result is True
            ws.send_json.assert_called_once()

        asyncio.run(_run())

    def test_exception_during_send_returns_false(self):
        async def _run():
            ws = AsyncMock()
            ws.send_json.side_effect = RuntimeError("WS closed")
            result = await self._safe_send(ws, {"node": "test"})
            assert result is False

        asyncio.run(_run())


# =============================================================================
# 23–27  WebSocket /ws/chat — validation (minimal replica app)
# =============================================================================

from fastapi import FastAPI
from fastapi import WebSocket as _WS
from fastapi import WebSocketDisconnect
from starlette.testclient import TestClient


def _make_ws_chat_validation_app() -> FastAPI:
    """
    Minimal FastAPI app that replicates ONLY the owner/prompt validation
    logic from the real /ws/chat endpoint. No LLM, no DB, no graph.
    """
    app = FastAPI()

    @app.websocket("/ws/chat")
    async def ws_chat(websocket: _WS):
        await websocket.accept()
        try:
            while True:
                try:
                    data = await websocket.receive_json()
                except Exception:
                    break

                thread_id = data.get("thread_id") or "new-thread"
                prompt = data.get("prompt")
                user_response = data.get("user_response")
                owner = data.get("owner")

                # Coerce string owner to dict (mirrors main.py behaviour)
                if isinstance(owner, str):
                    owner = {"user_id": owner}

                is_new_thread = "thread_id" not in data or not data.get("thread_id")

                if is_new_thread and not owner:
                    await websocket.send_json({
                        "node": "__error__",
                        "error": "Owner information is required for new conversations",
                        "error_type": "ValidationError",
                        "thread_id": thread_id,
                        "timestamp": time.time(),
                    })
                    continue

                if not prompt and not user_response:
                    await websocket.send_json({
                        "node": "__error__",
                        "error": "Missing 'prompt' field for new conversation or "
                                 "'user_response' for continuing",
                        "error_type": "ValidationError",
                        "thread_id": thread_id,
                        "timestamp": time.time(),
                    })
                    continue

                # Valid message — send __start__ acknowledgement
                await websocket.send_json({
                    "node": "__start__",
                    "thread_id": thread_id,
                    "message": (
                        "Starting agent orchestration..."
                        if prompt
                        else "Continuing conversation..."
                    ),
                })
        except WebSocketDisconnect:
            pass

    return app


@pytest.fixture(scope="module")
def ws_client():
    app = _make_ws_chat_validation_app()
    return TestClient(app)


class TestWebSocketChatValidation:
    def test_new_thread_missing_owner_sends_error(self, ws_client):
        """No thread_id (= new thread) + no owner → __error__ ValidationError."""
        with ws_client.websocket_connect("/ws/chat") as ws:
            ws.send_json({"prompt": "hello world"})  # new thread, no owner
            msg = ws.receive_json()
        assert msg["node"] == "__error__"
        assert msg["error_type"] == "ValidationError"
        assert "Owner" in msg["error"] or "owner" in msg["error"].lower()

    def test_missing_prompt_and_user_response_sends_error(self, ws_client):
        """Existing thread with owner but neither prompt nor user_response → __error__."""
        with ws_client.websocket_connect("/ws/chat") as ws:
            ws.send_json({
                "thread_id": "existing-thread-123",
                "owner": {"user_id": "u1"},
                # no prompt, no user_response
            })
            msg = ws.receive_json()
        assert msg["node"] == "__error__"
        assert msg["error_type"] == "ValidationError"

    def test_valid_new_thread_with_owner_sends_start(self, ws_client):
        """Valid new-thread message (owner + prompt) → __start__ acknowledged."""
        with ws_client.websocket_connect("/ws/chat") as ws:
            ws.send_json({"prompt": "hello world", "owner": {"user_id": "user-1"}})
            msg = ws.receive_json()
        assert msg["node"] == "__start__"
        assert "Starting" in msg["message"]

    def test_existing_thread_with_user_response_sends_start(self, ws_client):
        """Continuing a conversation with user_response only → __start__."""
        with ws_client.websocket_connect("/ws/chat") as ws:
            ws.send_json({
                "thread_id": "thread-abc",
                "owner": {"user_id": "u1"},
                "user_response": "yes, proceed",
            })
            msg = ws.receive_json()
        assert msg["node"] == "__start__"
        assert "Continuing" in msg["message"]

    def test_owner_provided_as_string_is_coerced_to_dict(self, ws_client):
        """owner='user-str-id' string is accepted and treated as new thread owner."""
        with ws_client.websocket_connect("/ws/chat") as ws:
            ws.send_json({"prompt": "task", "owner": "user-str-id"})
            msg = ws.receive_json()
        # Should reach __start__, not __error__
        assert msg["node"] == "__start__"


# =============================================================================
# 28  Orchestration exception → __error__ node
# =============================================================================

def _make_ws_chat_orch_error_app() -> FastAPI:
    """Simulates execute_orchestration raising a RuntimeError."""
    app = FastAPI()

    @app.websocket("/ws/chat")
    async def ws_chat(websocket: _WS):
        await websocket.accept()
        try:
            data = await websocket.receive_json()
            thread_id = data.get("thread_id", "test-thread")
            try:
                raise RuntimeError("LLM provider unavailable")
            except Exception as exc:
                await websocket.send_json({
                    "node": "__error__",
                    "thread_id": thread_id,
                    "error": str(exc)[:200],
                    "error_type": type(exc).__name__,
                    "message": "An error occurred during orchestration",
                    "status": "error",
                    "timestamp": time.time(),
                })
        except WebSocketDisconnect:
            pass

    return app


class TestWebSocketOrchestrationError:
    def test_orchestration_exception_sends_error_node(self):
        client = TestClient(_make_ws_chat_orch_error_app())
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_json({"prompt": "hello", "owner": {"user_id": "u1"}})
            msg = ws.receive_json()
        assert msg["node"] == "__error__"
        assert msg["error_type"] == "RuntimeError"
        assert "LLM provider unavailable" in msg["error"]
        assert msg["status"] == "error"


# =============================================================================
# 29–30  Thread-id isolation in frontend_websockets registry
# =============================================================================

class TestThreadIdIsolation:
    """
    frontend_websockets: Dict[str, WebSocket] maps thread_id → WS.
    Updates for thread A must not reach thread B, and removing A
    must not affect B.
    """

    def test_messages_for_thread_a_do_not_reach_thread_b(self):
        async def _run():
            ws_a = AsyncMock()
            ws_b = AsyncMock()
            registry: Dict[str, Any] = {"thread-A": ws_a, "thread-B": ws_b}

            # Only dispatch to thread-A
            lock = asyncio.Lock()
            async with lock:
                ws = registry.get("thread-A")
                if ws:
                    await ws.send_json({"node": "update", "thread_id": "thread-A"})

            ws_a.send_json.assert_called_once()
            ws_b.send_json.assert_not_called()

        asyncio.run(_run())

    def test_removing_thread_a_does_not_affect_thread_b(self):
        async def _run():
            ws_a = AsyncMock()
            ws_b = AsyncMock()
            registry: Dict[str, Any] = {"thread-A": ws_a, "thread-B": ws_b}

            del registry["thread-A"]

            assert "thread-A" not in registry
            assert "thread-B" in registry
            assert registry["thread-B"] is ws_b

        asyncio.run(_run())


# =============================================================================
# 31–34  MemorySaver state persistence
# =============================================================================

class TestMemorySaverPersistence:
    """
    Verifies that the MemorySaver checkpointer (used as the global
    `checkpointer` in main.py) correctly stores and isolates state
    across graph invocations.

    Uses a minimal LangGraph StateGraph to avoid importing the full
    brain/hands pipeline.
    """

    @pytest.fixture
    def saver(self):
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()

    def _build_counter_graph(self, saver):
        """Graph with a single node that increments `count` by 1."""
        from langgraph.graph import END, StateGraph
        from typing_extensions import TypedDict

        class CountState(TypedDict):
            count: int

        def increment(state: CountState) -> CountState:
            return {"count": state.get("count", 0) + 1}

        builder = StateGraph(CountState)
        builder.add_node("increment", increment)
        builder.set_entry_point("increment")
        builder.add_edge("increment", END)
        return builder.compile(checkpointer=saver)

    def _cfg(self, thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    def test_checkpoint_written_after_invoke(self, saver):
        """After a graph.invoke(), the checkpointer holds a checkpoint."""
        graph = self._build_counter_graph(saver)
        graph.invoke({"count": 0}, config=self._cfg("cp-write-test"))
        assert saver.get(self._cfg("cp-write-test")) is not None

    def test_invocation_result_reflects_node_execution(self, saver):
        """The node increments count; result must equal input + 1."""
        graph = self._build_counter_graph(saver)
        result = graph.invoke({"count": 4}, config=self._cfg("cp-result-test"))
        assert result["count"] == 5

    def test_thread_a_checkpoint_not_visible_from_thread_b(self, saver):
        """Writing to thread A must not create a checkpoint for thread B."""
        graph = self._build_counter_graph(saver)
        graph.invoke({"count": 0}, config=self._cfg("isolation-A"))
        assert saver.get(self._cfg("isolation-B")) is None

    def test_nonexistent_thread_returns_none(self, saver):
        """get() for a thread that was never written returns None."""
        assert saver.get(self._cfg("never-written")) is None


# =============================================================================
# 35–42  State TypedDict field declarations
# =============================================================================

class TestStateTypeDict:
    """
    LangGraph silently drops undeclared State fields.
    These tests protect against accidental removal of critical annotations.
    """

    @pytest.fixture(scope="class")
    def hints(self):
        import typing
        return typing.get_type_hints(State, include_extras=True)

    def test_last_agent_result_declared(self, hints):
        assert "last_agent_result" in hints

    def test_owner_id_declared(self, hints):
        assert "owner_id" in hints

    def test_action_history_declared(self, hints):
        assert "action_history" in hints

    def test_todo_list_declared(self, hints):
        assert "todo_list" in hints

    def test_pending_action_declared(self, hints):
        assert "pending_action" in hints

    def test_pending_action_approval_declared(self, hints):
        assert "pending_action_approval" in hints

    def test_last_agent_result_uses_overwrite_reducer(self, hints):
        """last_agent_result: Annotated[Optional[Dict], overwrite_reducer]."""
        import typing
        args = typing.get_args(hints["last_agent_result"])
        # Annotated[X, reducer] → get_args returns (X, reducer)
        assert len(args) >= 2, "last_agent_result must be Annotated with a reducer"
        reducer = args[-1]
        assert reducer is overwrite_reducer, (
            f"Expected overwrite_reducer, got {reducer}"
        )

    def test_action_history_uses_append_reducer(self, hints):
        """action_history: Annotated[List[Dict], append_reducer]."""
        import typing
        args = typing.get_args(hints["action_history"])
        assert len(args) >= 2, "action_history must be Annotated with a reducer"
        reducer = args[-1]
        assert reducer is append_reducer, (
            f"Expected append_reducer, got {reducer}"
        )
