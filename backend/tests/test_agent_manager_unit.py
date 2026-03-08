"""
Unit tests for backend/services/agent_manager.py

User-specified critical tests (all covered):
  - test_agent_spawned_on_correct_port
  - test_adopt_existing_process_on_default_port
  - test_no_double_spawn_if_already_running
  - test_health_check_marks_agent_healthy / unhealthy
  - test_execute_posts_to_agent_endpoint
  - test_streaming_execute_relays_sse_events
  - test_agent_restart_on_crash
  - test_port_pool_no_conflicts

Full coverage:
  _ExternalProcess sentinel, PortPool (allocate/release/conflicts),
  ProcessManager (start/stop/is_running), HealthChecker (check/wait),
  AutoTerminator (idle detection), _get_agent_timeout (env override + table),
  AgentManager (spawn all paths, execute, streaming, terminate, is_active),
  get_agent_manager singleton.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent        # backend/
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent))                 # project root

from backend.services.agent_manager import (
    DEFAULT_AGENT_EXECUTE_TIMEOUT,
    AGENT_EXECUTE_TIMEOUTS,
    AgentInstance,
    AgentManager,
    AutoTerminator,
    HealthChecker,
    PortPool,
    ProcessManager,
    _ExternalProcess,
    _get_agent_timeout,
    get_agent_manager,
)
import backend.services.agent_manager as am_module
import time


# ── Fixtures / helpers ────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the global AgentManager singleton between tests."""
    orig = am_module._agent_manager
    am_module._agent_manager = None
    yield
    am_module._agent_manager = orig


def _make_instance(agent_id="spreadsheet_agent", port=9000, healthy=True):
    """Build a fake AgentInstance backed by an _ExternalProcess (always alive)."""
    return AgentInstance(
        agent_id=agent_id,
        process=_ExternalProcess(),
        port=port,
        pid=-1,
        start_time=time.time(),
        healthy=healthy,
    )


def _mock_httpx_post(json_data, status_code=200):
    """Return a mock httpx.AsyncClient context that responds to .post()."""
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.json.return_value = json_data
    mock_response.raise_for_status = MagicMock()

    mock_inner = MagicMock()
    mock_inner.post = AsyncMock(return_value=mock_response)

    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_inner)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    return mock_ctx, mock_inner


def _mock_sse_stream(lines):
    """Build an httpx mock that streams the given SSE text lines."""

    async def _aiter():
        for line in lines:
            yield line

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.aiter_lines = _aiter          # returns an async generator

    mock_stream_ctx = MagicMock()
    mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
    mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_inner = MagicMock()
    mock_inner.stream = MagicMock(return_value=mock_stream_ctx)

    mock_client_ctx = MagicMock()
    mock_client_ctx.__aenter__ = AsyncMock(return_value=mock_inner)
    mock_client_ctx.__aexit__ = AsyncMock(return_value=False)
    return mock_client_ctx, mock_inner


# ─────────────────────────────────────────────────────────────────────────────
# _ExternalProcess sentinel
# ─────────────────────────────────────────────────────────────────────────────

class TestExternalProcess:
    def test_poll_always_returns_none(self):
        ep = _ExternalProcess()
        assert ep.poll() is None

    def test_terminate_is_noop(self):
        ep = _ExternalProcess()
        ep.terminate()       # must not raise

    def test_kill_is_noop(self):
        ep = _ExternalProcess()
        ep.kill()            # must not raise

    def test_send_signal_is_noop(self):
        ep = _ExternalProcess()
        ep.send_signal(9)    # must not raise

    def test_wait_returns_zero(self):
        ep = _ExternalProcess()
        assert ep.wait() == 0

    def test_pid_is_negative_one(self):
        assert _ExternalProcess().pid == -1


# ─────────────────────────────────────────────────────────────────────────────
# _get_agent_timeout
# ─────────────────────────────────────────────────────────────────────────────

class TestGetAgentTimeout:
    def test_known_agent_returns_table_value(self):
        assert _get_agent_timeout("browser_agent") == AGENT_EXECUTE_TIMEOUTS["browser_agent"]
        assert _get_agent_timeout("coding_agent") == 600.0
        assert _get_agent_timeout("spreadsheet_agent") == 180.0

    def test_unknown_agent_returns_default(self):
        assert _get_agent_timeout("nonexistent_agent") == DEFAULT_AGENT_EXECUTE_TIMEOUT

    def test_env_override_takes_precedence(self):
        with patch.dict("os.environ", {"AGENT_TIMEOUT_MY_AGENT": "999"}):
            assert _get_agent_timeout("my_agent") == 999.0

    def test_invalid_env_override_falls_back_to_table(self):
        with patch.dict("os.environ", {"AGENT_TIMEOUT_SPREADSHEET_AGENT": "not_a_number"}):
            assert _get_agent_timeout("spreadsheet_agent") == 180.0

    def test_browser_automation_agent_maps_correctly(self):
        """SKILL.md uses browser_automation_agent — must have 900s timeout."""
        assert _get_agent_timeout("browser_automation_agent") == 900.0


# ─────────────────────────────────────────────────────────────────────────────
# PortPool
# ─────────────────────────────────────────────────────────────────────────────

class TestPortPool:
    @pytest.mark.asyncio
    async def test_default_port_allocated_when_free(self):
        pool = PortPool()
        with patch.object(pool, "_is_port_in_use", return_value=False):
            port = await pool.allocate("spreadsheet_agent")
        assert port == PortPool.DEFAULT_PORTS["spreadsheet_agent"]  # 9000

    @pytest.mark.asyncio
    async def test_dynamic_port_used_when_default_taken(self):
        pool = PortPool()
        # Default port is busy; all dynamic range starts free
        def _in_use(p):
            return p == PortPool.DEFAULT_PORTS["spreadsheet_agent"]
        with patch.object(pool, "_is_port_in_use", side_effect=_in_use):
            port = await pool.allocate("spreadsheet_agent")
        assert port >= PortPool.DYNAMIC_RANGE[0]

    @pytest.mark.asyncio
    async def test_already_allocated_returns_same_port(self):
        pool = PortPool()
        with patch.object(pool, "_is_port_in_use", return_value=False):
            p1 = await pool.allocate("document_agent")
            p2 = await pool.allocate("document_agent")
        assert p1 == p2

    @pytest.mark.asyncio
    async def test_two_agents_no_port_conflict(self):
        """CRITICAL: test_port_pool_no_conflicts — different agents get different ports."""
        pool = PortPool()
        with patch.object(pool, "_is_port_in_use", return_value=False):
            port_a = await pool.allocate("spreadsheet_agent")
            port_b = await pool.allocate("document_agent")
        assert port_a != port_b

    @pytest.mark.asyncio
    async def test_release_frees_port(self):
        pool = PortPool()
        with patch.object(pool, "_is_port_in_use", return_value=False):
            port = await pool.allocate("universal_agent")
        await pool.release("universal_agent")
        assert "universal_agent" not in pool.allocated_ports
        assert port not in pool.used_ports

    @pytest.mark.asyncio
    async def test_no_available_ports_raises(self):
        pool = PortPool()
        # All ports in use
        with patch.object(pool, "_is_port_in_use", return_value=True):
            with pytest.raises(RuntimeError, match="No available ports"):
                await pool.allocate("unknown_dynamic_agent")


# ─────────────────────────────────────────────────────────────────────────────
# ProcessManager
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessManager:
    def test_unknown_agent_raises_value_error(self, tmp_path):
        pm = ProcessManager(tmp_path)
        with pytest.raises(ValueError, match="Unknown agent"):
            asyncio.get_event_loop().run_until_complete(
                pm.start_agent("nonexistent_agent", 9999)
            )

    @pytest.mark.asyncio
    async def test_start_agent_uses_uvicorn_when_init_exists(self, tmp_path):
        pm = ProcessManager(tmp_path)
        agent_id = "spreadsheet_agent"
        module_path = ProcessManager.AGENT_MODULE_MAP[agent_id]
        agent_dir = tmp_path / module_path.replace(".", "/")
        agent_dir.mkdir(parents=True)
        (agent_dir / "__init__.py").write_text("# init")

        mock_proc = MagicMock()
        mock_proc.pid = 12345
        with patch("backend.services.agent_manager.subprocess.Popen", return_value=mock_proc) as mock_popen:
            process = await pm.start_agent(agent_id, 9000)

        cmd = mock_popen.call_args[0][0]
        assert "uvicorn" in cmd
        assert f"{module_path}.__init__:app" in cmd
        assert "9000" in cmd

    @pytest.mark.asyncio
    async def test_start_agent_sets_env_variables(self, tmp_path):
        pm = ProcessManager(tmp_path)
        agent_id = "document_agent"
        module_path = ProcessManager.AGENT_MODULE_MAP[agent_id]
        agent_dir = tmp_path / module_path.replace(".", "/")
        agent_dir.mkdir(parents=True)
        (agent_dir / "__init__.py").write_text("# init")

        mock_proc = MagicMock()
        mock_proc.pid = 99
        with patch("backend.services.agent_manager.subprocess.Popen", return_value=mock_proc) as mock_popen:
            await pm.start_agent(agent_id, 8050)

        env = mock_popen.call_args[1]["env"]
        assert env["AGENT_PORT"] == "8050"
        assert env["AGENT_ID"] == agent_id

    def test_is_running_true_when_poll_returns_none(self):
        pm = ProcessManager(Path("."))
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        assert pm.is_running(mock_proc) is True

    def test_is_running_false_when_poll_returns_exit_code(self):
        pm = ProcessManager(Path("."))
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        assert pm.is_running(mock_proc) is False

    @pytest.mark.asyncio
    async def test_stop_agent_returns_true_if_already_stopped(self):
        pm = ProcessManager(Path("."))
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1   # already stopped
        result = await pm.stop_agent(mock_proc)
        assert result is True
        mock_proc.terminate.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_agent_terminates_running_process(self):
        pm = ProcessManager(Path("."))
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None       # running
        mock_proc.wait = MagicMock()             # wait doesn't time out
        import os as _os
        with patch("os.name", "posix"):
            result = await pm.stop_agent(mock_proc, timeout=5)
        assert result is True
        mock_proc.terminate.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_agent_force_kills_on_timeout(self):
        import subprocess as _subprocess
        pm = ProcessManager(Path("."))
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.wait.side_effect = [_subprocess.TimeoutExpired(cmd="x", timeout=1), None]
        with patch("os.name", "posix"):
            result = await pm.stop_agent(mock_proc, timeout=1)
        assert result is True
        mock_proc.kill.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# HealthChecker
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthChecker:
    @pytest.mark.asyncio
    async def test_check_health_returns_true_on_200(self):
        hc = HealthChecker()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_inner = AsyncMock()
        mock_inner.get = AsyncMock(return_value=mock_response)
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_inner)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        with patch("backend.services.agent_manager.httpx.AsyncClient", return_value=mock_ctx):
            result = await hc.check_health(9000)
        assert result is True

    @pytest.mark.asyncio
    async def test_check_health_returns_false_on_connect_error(self):
        import httpx as _httpx
        hc = HealthChecker()
        mock_inner = AsyncMock()
        mock_inner.get = AsyncMock(side_effect=_httpx.ConnectError("refused"))
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_inner)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        with patch("backend.services.agent_manager.httpx.AsyncClient", return_value=mock_ctx):
            result = await hc.check_health(9000)
        assert result is False

    @pytest.mark.asyncio
    async def test_check_health_returns_false_on_non_200(self):
        hc = HealthChecker()
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_inner = AsyncMock()
        mock_inner.get = AsyncMock(return_value=mock_response)
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_inner)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        with patch("backend.services.agent_manager.httpx.AsyncClient", return_value=mock_ctx):
            result = await hc.check_health(9000)
        assert result is False

    @pytest.mark.asyncio
    async def test_wait_for_ready_returns_true_on_first_healthy_poll(self):
        hc = HealthChecker(timeout=5)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_inner = AsyncMock()
        mock_inner.get = AsyncMock(return_value=mock_response)
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_inner)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        with patch("backend.services.agent_manager.httpx.AsyncClient", return_value=mock_ctx):
            result = await hc.wait_for_ready(9000, "spreadsheet_agent")
        assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_ready_returns_false_on_timeout(self):
        import httpx as _httpx
        hc = HealthChecker(timeout=1)
        hc.check_interval = 0.1
        mock_inner = AsyncMock()
        mock_inner.get = AsyncMock(side_effect=_httpx.ConnectError("down"))
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_inner)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        with patch("backend.services.agent_manager.httpx.AsyncClient", return_value=mock_ctx):
            result = await hc.wait_for_ready(9000, "spreadsheet_agent")
        assert result is False

    def test_get_timeout_per_agent(self):
        hc = HealthChecker()
        assert hc._get_timeout("coding_agent") == 120
        assert hc._get_timeout("document_agent") == 90
        assert hc._get_timeout("unknown_agent") == hc.default_timeout


# ─────────────────────────────────────────────────────────────────────────────
# AutoTerminator
# ─────────────────────────────────────────────────────────────────────────────

class TestAutoTerminator:
    @pytest.mark.asyncio
    async def test_idle_agent_marked_for_termination(self):
        am = AgentManager()
        at = AutoTerminator(am, idle_timeout=10)

        old_instance = _make_instance("spreadsheet_agent", 9000)
        old_instance.last_used = time.time() - 100   # idle for 100s > 10s threshold
        am.active_agents["spreadsheet_agent"] = old_instance

        with patch.object(am, "terminate_agent", new=AsyncMock(return_value=True)) as mock_term:
            await at._check_idle_agents()

        mock_term.assert_called_once_with("spreadsheet_agent")

    @pytest.mark.asyncio
    async def test_active_agent_not_terminated(self):
        am = AgentManager()
        at = AutoTerminator(am, idle_timeout=300)

        fresh_instance = _make_instance("spreadsheet_agent", 9000)
        fresh_instance.last_used = time.time()   # just used
        am.active_agents["spreadsheet_agent"] = fresh_instance

        with patch.object(am, "terminate_agent", new=AsyncMock()) as mock_term:
            await at._check_idle_agents()

        mock_term.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self):
        am = AgentManager()
        at = AutoTerminator(am, idle_timeout=300)
        await at.start_monitoring()
        assert at._monitoring is True
        await at.stop_monitoring()
        assert at._monitoring is False


# ─────────────────────────────────────────────────────────────────────────────
# AgentManager — spawn_agent
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentManagerSpawn:

    def _make_am(self):
        am = AgentManager()
        am._initialized = True
        return am

    @pytest.mark.asyncio
    async def test_agent_spawned_on_correct_port(self):
        """CRITICAL: spawned instance uses the port returned by PortPool.allocate."""
        am = self._make_am()
        mock_proc = MagicMock()
        mock_proc.pid = 1234
        mock_proc.poll.return_value = None

        with patch.object(am.port_pool, "allocate", new=AsyncMock(return_value=9042)):
            with patch.object(am.health_checker, "check_health", new=AsyncMock(return_value=False)):
                with patch.object(am.process_manager, "start_agent", new=AsyncMock(return_value=mock_proc)):
                    with patch.object(am.health_checker, "wait_for_ready", new=AsyncMock(return_value=True)):
                        instance = await am.spawn_agent("spreadsheet_agent")

        assert instance.port == 9042
        assert instance.healthy is True
        assert instance.pid == 1234

    @pytest.mark.asyncio
    async def test_adopt_existing_process_on_default_port(self):
        """CRITICAL: agent already running on default port is adopted, not re-spawned."""
        am = self._make_am()
        browser_default = PortPool.DEFAULT_PORTS["browser_agent"]

        with patch.object(am.health_checker, "check_health", new=AsyncMock(return_value=True)):
            with patch.object(am.process_manager, "start_agent", new=AsyncMock()) as mock_start:
                instance = await am.spawn_agent("browser_agent")

        mock_start.assert_not_called()
        assert instance.port == browser_default
        assert isinstance(instance.process, _ExternalProcess)
        assert instance.healthy is True

    @pytest.mark.asyncio
    async def test_no_double_spawn_if_already_running(self):
        """CRITICAL: second spawn_agent call returns existing instance without new process."""
        am = self._make_am()
        mock_proc = MagicMock()
        mock_proc.pid = 111
        mock_proc.poll.return_value = None

        with patch.object(am.port_pool, "allocate", new=AsyncMock(return_value=9001)):
            with patch.object(am.health_checker, "check_health", new=AsyncMock(return_value=False)):
                with patch.object(am.process_manager, "start_agent", new=AsyncMock(return_value=mock_proc)) as mock_start:
                    with patch.object(am.health_checker, "wait_for_ready", new=AsyncMock(return_value=True)):
                        inst1 = await am.spawn_agent("universal_agent")
                        inst2 = await am.spawn_agent("universal_agent")

        assert inst1 is inst2
        mock_start.assert_called_once()   # only spawned once

    @pytest.mark.asyncio
    async def test_agent_restart_on_crash(self):
        """CRITICAL: when tracked process has exited, spawn_agent respawns it."""
        am = self._make_am()

        # Pre-populate with a "crashed" process (poll returns non-None)
        crashed_proc = MagicMock()
        crashed_proc.poll.return_value = 1      # exit code 1 = crashed
        crashed_proc.pid = 999
        am.active_agents["spreadsheet_agent"] = AgentInstance(
            agent_id="spreadsheet_agent",
            process=crashed_proc,
            port=9000,
            pid=999,
            start_time=time.time() - 60,
            healthy=True,
        )
        am.port_pool.allocated_ports["spreadsheet_agent"] = 9000
        am.port_pool.used_ports.add(9000)

        new_proc = MagicMock()
        new_proc.pid = 2222
        new_proc.poll.return_value = None

        with patch.object(am.health_checker, "check_health", new=AsyncMock(return_value=False)):
            with patch.object(am.port_pool, "allocate", new=AsyncMock(return_value=9002)):
                with patch.object(am.process_manager, "start_agent", new=AsyncMock(return_value=new_proc)) as mock_start:
                    with patch.object(am.health_checker, "wait_for_ready", new=AsyncMock(return_value=True)):
                        with patch.object(am.process_manager, "stop_agent", new=AsyncMock(return_value=True)):
                            instance = await am.spawn_agent("spreadsheet_agent")

        mock_start.assert_called_once()    # re-spawned
        assert instance.pid == 2222       # new process
        assert instance.port == 9002

    @pytest.mark.asyncio
    async def test_spawn_cleans_up_on_health_failure(self):
        """When wait_for_ready returns False, port is released and exception raised."""
        am = self._make_am()
        mock_proc = MagicMock()
        mock_proc.pid = 500
        mock_proc.poll.return_value = None

        with patch.object(am.health_checker, "check_health", new=AsyncMock(return_value=False)):
            with patch.object(am.port_pool, "allocate", new=AsyncMock(return_value=9010)):
                with patch.object(am.process_manager, "start_agent", new=AsyncMock(return_value=mock_proc)):
                    with patch.object(am.health_checker, "wait_for_ready", new=AsyncMock(return_value=False)):
                        with patch.object(am.port_pool, "release", new=AsyncMock()) as mock_release:
                            with patch.object(am.process_manager, "stop_agent", new=AsyncMock()):
                                with pytest.raises(RuntimeError, match="health check"):
                                    await am.spawn_agent("spreadsheet_agent")

        mock_release.assert_called_once_with("spreadsheet_agent")


# ─────────────────────────────────────────────────────────────────────────────
# AgentManager — execute (non-streaming)
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentManagerExecute:

    def _make_am_with_agent(self, agent_id="spreadsheet_agent", port=9000):
        am = AgentManager()
        am._initialized = True
        instance = _make_instance(agent_id, port)
        am.active_agents[agent_id] = instance
        return am, instance

    @pytest.mark.asyncio
    async def test_execute_posts_to_agent_endpoint(self):
        """CRITICAL: execute() POSTs to http://localhost:{port}/execute."""
        am, _ = self._make_am_with_agent("spreadsheet_agent", 9000)
        expected_response = {"status": "ok", "result": {"answer": "42"}}
        mock_ctx, mock_inner = _mock_httpx_post(expected_response)

        with patch("backend.services.agent_manager.httpx.AsyncClient", return_value=mock_ctx):
            result = await am.execute("spreadsheet_agent", {"prompt": "sum column A"})

        post_url = mock_inner.post.call_args[0][0]
        assert post_url == "http://localhost:9000/execute"
        assert result == expected_response

    @pytest.mark.asyncio
    async def test_execute_includes_prompt_in_request(self):
        am, _ = self._make_am_with_agent("universal_agent", 8070)
        mock_ctx, mock_inner = _mock_httpx_post({"status": "ok"})

        with patch("backend.services.agent_manager.httpx.AsyncClient", return_value=mock_ctx):
            await am.execute("universal_agent", {"prompt": "hello world", "thread_id": "t1"})

        body = mock_inner.post.call_args[1]["json"]
        assert body["prompt"] == "hello world"
        assert body["thread_id"] == "t1"

    @pytest.mark.asyncio
    async def test_execute_timeout_returns_error_dict(self):
        import httpx as _httpx
        am, _ = self._make_am_with_agent("browser_agent", 8090)
        mock_inner = MagicMock()
        mock_inner.post = AsyncMock(side_effect=_httpx.TimeoutException("timed out"))
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_inner)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("backend.services.agent_manager.httpx.AsyncClient", return_value=mock_ctx):
            result = await am.execute("browser_agent", {"prompt": "browse"})

        assert result["status"] == "error"
        assert "timed out" in result["error"].lower() or "browser_agent" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_http_error_returns_error_dict(self):
        import httpx as _httpx
        am, _ = self._make_am_with_agent("document_agent", 8050)
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_inner = MagicMock()
        mock_inner.post = AsyncMock(
            side_effect=_httpx.HTTPStatusError("error", request=MagicMock(), response=mock_response)
        )
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_inner)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("backend.services.agent_manager.httpx.AsyncClient", return_value=mock_ctx):
            result = await am.execute("document_agent", {"prompt": "parse"})

        assert result["status"] == "error"
        assert "500" in result["error"] or "document_agent" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_uses_per_agent_timeout(self):
        """Browser agent gets 900s timeout, not the 120s default."""
        am, _ = self._make_am_with_agent("browser_agent", 8090)
        mock_ctx, mock_inner = _mock_httpx_post({"status": "ok"})

        with patch("backend.services.agent_manager.httpx.AsyncClient", return_value=mock_ctx):
            await am.execute("browser_agent", {"prompt": "browse"})

        timeout_used = mock_inner.post.call_args[1]["timeout"]
        assert timeout_used == 900.0

    @pytest.mark.asyncio
    async def test_execute_sends_user_id_header(self):
        am, _ = self._make_am_with_agent("spreadsheet_agent", 9000)
        mock_ctx, mock_inner = _mock_httpx_post({"status": "ok"})

        with patch("backend.services.agent_manager.httpx.AsyncClient", return_value=mock_ctx):
            await am.execute("spreadsheet_agent", {"prompt": "go", "user_id": "user-123"})

        headers = mock_inner.post.call_args[1]["headers"]
        assert headers["X-User-ID"] == "user-123"

    @pytest.mark.asyncio
    async def test_execute_updates_last_used_on_success(self):
        am, instance = self._make_am_with_agent("spreadsheet_agent", 9000)
        old_ts = instance.last_used - 10
        instance.last_used = old_ts
        mock_ctx, _ = _mock_httpx_post({"status": "ok"})

        with patch("backend.services.agent_manager.httpx.AsyncClient", return_value=mock_ctx):
            await am.execute("spreadsheet_agent", {"prompt": "go"})

        assert instance.last_used > old_ts


# ─────────────────────────────────────────────────────────────────────────────
# AgentManager — _execute_streaming (SSE)
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentManagerStreaming:

    @pytest.mark.asyncio
    async def test_streaming_execute_relays_sse_events(self):
        """CRITICAL: progress_callback called for each 'progress' SSE event."""
        am = AgentManager()
        am._initialized = True

        sse_lines = [
            'data: {"type": "progress", "message": "step 1"}',
            'data: {"type": "progress", "message": "step 2"}',
            'data: {"type": "done", "result": {"answer": "done"}}',
        ]
        mock_ctx, _ = _mock_sse_stream(sse_lines)
        received = []

        async def cb(msg):
            received.append(msg)

        with patch("backend.services.agent_manager.httpx.AsyncClient", return_value=mock_ctx):
            result = await am._execute_streaming(
                "http://localhost:9000/execute/stream",
                {"prompt": "go"},
                {},
                120.0,
                cb,
            )

        assert received == ["step 1", "step 2"]
        assert result == {"answer": "done"}

    @pytest.mark.asyncio
    async def test_streaming_returns_done_result(self):
        am = AgentManager()
        sse_lines = [
            'data: {"type": "done", "result": {"rows": 42}}',
        ]
        mock_ctx, _ = _mock_sse_stream(sse_lines)

        with patch("backend.services.agent_manager.httpx.AsyncClient", return_value=mock_ctx):
            result = await am._execute_streaming(
                "http://localhost:9000/execute/stream",
                {}, {}, 120.0, AsyncMock()
            )

        assert result == {"rows": 42}

    @pytest.mark.asyncio
    async def test_streaming_error_event_raises(self):
        am = AgentManager()
        sse_lines = [
            'data: {"type": "error", "message": "agent crashed"}',
        ]
        mock_ctx, _ = _mock_sse_stream(sse_lines)

        with patch("backend.services.agent_manager.httpx.AsyncClient", return_value=mock_ctx):
            with pytest.raises(RuntimeError, match="agent crashed"):
                await am._execute_streaming(
                    "http://localhost:9000/execute/stream",
                    {}, {}, 120.0, AsyncMock()
                )

    @pytest.mark.asyncio
    async def test_streaming_no_done_event_raises(self):
        """Stream ends without 'done' → RuntimeError."""
        am = AgentManager()
        sse_lines = [
            'data: {"type": "progress", "message": "working..."}',
            # no done event
        ]
        mock_ctx, _ = _mock_sse_stream(sse_lines)

        with patch("backend.services.agent_manager.httpx.AsyncClient", return_value=mock_ctx):
            with pytest.raises(RuntimeError, match="done"):
                await am._execute_streaming(
                    "http://localhost:9000/execute/stream",
                    {}, {}, 120.0, AsyncMock()
                )

    @pytest.mark.asyncio
    async def test_streaming_heartbeat_lines_ignored(self):
        """Lines not starting with 'data: ' (heartbeats) are silently skipped."""
        am = AgentManager()
        sse_lines = [
            ": heartbeat",
            "",
            'data: {"type": "done", "result": {"ok": true}}',
        ]
        mock_ctx, _ = _mock_sse_stream(sse_lines)
        cb = AsyncMock()

        with patch("backend.services.agent_manager.httpx.AsyncClient", return_value=mock_ctx):
            result = await am._execute_streaming(
                "http://localhost:9000/execute/stream",
                {}, {}, 120.0, cb
            )

        assert result == {"ok": True}
        cb.assert_not_called()    # no progress events

    @pytest.mark.asyncio
    async def test_execute_falls_back_to_regular_on_streaming_error(self):
        """If streaming raises, execute() falls back to /execute endpoint."""
        am = AgentManager()
        am._initialized = True
        instance = _make_instance("spreadsheet_agent", 9000)
        am.active_agents["spreadsheet_agent"] = instance

        fallback_response = {"status": "ok", "result": "fallback"}
        mock_regular_ctx, _ = _mock_httpx_post(fallback_response)

        with patch.object(am, "_execute_streaming", new=AsyncMock(side_effect=Exception("stream failed"))):
            with patch("backend.services.agent_manager.httpx.AsyncClient", return_value=mock_regular_ctx):
                result = await am.execute(
                    "spreadsheet_agent",
                    {"prompt": "go"},
                    progress_callback=AsyncMock(),
                )

        assert result == fallback_response


# ─────────────────────────────────────────────────────────────────────────────
# AgentManager — terminate / cleanup
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentManagerTerminate:
    @pytest.mark.asyncio
    async def test_terminate_agent_returns_true_and_cleans_up(self):
        am = AgentManager()
        am._initialized = True
        am.active_agents["spreadsheet_agent"] = _make_instance("spreadsheet_agent", 9000)
        am.port_pool.allocated_ports["spreadsheet_agent"] = 9000
        am.port_pool.used_ports.add(9000)

        with patch.object(am.process_manager, "stop_agent", new=AsyncMock(return_value=True)):
            result = await am.terminate_agent("spreadsheet_agent")

        assert result is True
        assert "spreadsheet_agent" not in am.active_agents
        assert "spreadsheet_agent" not in am.port_pool.allocated_ports

    @pytest.mark.asyncio
    async def test_terminate_nonexistent_agent_returns_false(self):
        am = AgentManager()
        result = await am.terminate_agent("unknown_agent")
        assert result is False

    @pytest.mark.asyncio
    async def test_terminate_all_clears_all_agents(self):
        am = AgentManager()
        am._initialized = True
        for agent_id in ["spreadsheet_agent", "document_agent", "universal_agent"]:
            am.active_agents[agent_id] = _make_instance(agent_id, 9000)

        with patch.object(am.process_manager, "stop_agent", new=AsyncMock(return_value=True)):
            await am.terminate_all()

        assert am.active_agents == {}

    @pytest.mark.asyncio
    async def test_terminate_all_continues_on_individual_error(self):
        am = AgentManager()
        am._initialized = True
        am.active_agents["a1"] = _make_instance("a1", 9001)
        am.active_agents["a2"] = _make_instance("a2", 9002)

        call_count = 0

        async def flaky_stop(process, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("stop failed")
            return True

        with patch.object(am.process_manager, "stop_agent", side_effect=flaky_stop):
            await am.terminate_all()   # must not raise


# ─────────────────────────────────────────────────────────────────────────────
# AgentManager — is_agent_active / get_active_agents
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentManagerMisc:
    def test_is_agent_active_true_when_running(self):
        am = AgentManager()
        am.active_agents["spreadsheet_agent"] = _make_instance("spreadsheet_agent", 9000)
        assert am.is_agent_active("spreadsheet_agent") is True

    def test_is_agent_active_false_when_not_tracked(self):
        am = AgentManager()
        assert am.is_agent_active("nonexistent") is False

    def test_is_agent_active_false_when_process_dead(self):
        am = AgentManager()
        dead_proc = MagicMock()
        dead_proc.poll.return_value = 1    # process exited
        am.active_agents["spreadsheet_agent"] = AgentInstance(
            agent_id="spreadsheet_agent",
            process=dead_proc,
            port=9000,
            pid=777,
            start_time=time.time(),
        )
        assert am.is_agent_active("spreadsheet_agent") is False

    def test_get_active_agents_returns_copy(self):
        am = AgentManager()
        am.active_agents["a"] = _make_instance("a", 9001)
        snapshot = am.get_active_agents()
        snapshot["b"] = _make_instance("b", 9002)
        # modifying snapshot does not affect internal dict
        assert "b" not in am.active_agents

    @pytest.mark.asyncio
    async def test_initialize_starts_auto_terminator(self):
        am = AgentManager()
        with patch.object(am.auto_terminator, "start_monitoring", new=AsyncMock()) as mock_start:
            await am.initialize()
        mock_start.assert_called_once()
        assert am._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self):
        am = AgentManager()
        with patch.object(am.auto_terminator, "start_monitoring", new=AsyncMock()) as mock_start:
            await am.initialize()
            await am.initialize()   # second call should be a no-op
        mock_start.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# get_agent_manager singleton
# ─────────────────────────────────────────────────────────────────────────────

class TestGetAgentManagerSingleton:
    def test_same_instance_on_repeated_calls(self):
        s1 = get_agent_manager()
        s2 = get_agent_manager()
        assert s1 is s2

    def test_returns_agent_manager_instance(self):
        assert isinstance(get_agent_manager(), AgentManager)
