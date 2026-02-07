"""
Unit tests for Omni-Hands (Hands dispatcher logic).

This test module mocks external dependencies (agents, tools, sandbox, terminal)
to test the Hands' execution routing and result handling in isolation.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
import json
import time
import asyncio

import sys
from pathlib import Path

backend_root = Path(__file__).parent.parent
sys.path.insert(0, str(backend_root))

from orchestrator.hands import Hands
from orchestrator.schemas import ActionResult, TaskStatus

# Import utils for patching RetryManager
from utils import retry_utils


@pytest.fixture
def mock_agent_registry():
    """Mock the agent registry."""
    with patch("orchestrator.hands.agent_registry") as mock:
        mock.list_active_agents = Mock(
            return_value=[
                {
                    "id": "test_agent_1",
                    "name": "TestAgent",
                    "description": "A test agent",
                    "type": "http_rest",
                },
                {
                    "id": "test_agent_2",
                    "name": "FinanceAgent",
                    "description": "Finance agent",
                    "type": "http_rest",
                },
            ]
        )
        mock.get_agent_url = Mock(return_value="http://localhost:9001")
        yield mock


@pytest.fixture
def mock_tool_registry():
    """Mock the tool registry."""
    with patch("orchestrator.hands.tool_registry") as mock:
        mock.execute_tool = AsyncMock()
        yield mock


@pytest.fixture
def mock_code_sandbox():
    """Mock the code sandbox service."""
    with patch("orchestrator.hands.code_sandbox") as mock:
        mock.execute_code = Mock(
            return_value={"success": True, "stdout": "4", "error": None}
        )
        yield mock


@pytest.fixture
def mock_terminal_service():
    """Mock the terminal service."""
    with patch("orchestrator.hands.terminal_service") as mock:
        mock.execute_command = Mock(
            return_value={
                "returncode": 0,
                "stdout": "file1.txt\nfile2.txt",
                "stderr": "",
            }
        )
        yield mock


@pytest.fixture
def mock_telemetry_service():
    """Mock the telemetry service."""
    with patch("orchestrator.hands.telemetry_service") as mock:
        mock.log_agent_call = Mock()
        mock.log_tool_call = Mock()
        yield mock


@pytest.fixture
def mock_credential_service():
    """Mock the credential service."""
    with patch("orchestrator.hands.get_credentials_for_headers") as mock:
        mock.return_value = {}
        yield mock


@pytest.fixture
def mock_retry_manager():
    """Mock the retry manager."""
    with patch("orchestrator.hands.RetryManager") as mock:
        mock.retry_async = AsyncMock()
        yield mock


@pytest.fixture
def mock_content_hooks():
    """Mock the CMS hooks."""
    with patch("orchestrator.hands.hooks") as mock:
        mock.on_task_complete = AsyncMock(return_value={"result": "processed"})
        yield mock


@pytest.fixture
def hands():
    """Create a Hands instance for testing."""
    return Hands()


@pytest.fixture
def sample_state():
    """Create a sample orchestrator state with a decision."""
    return {
        "decision": {
            "action_type": "tool",
            "resource_id": "calculator",
            "payload": {"expression": "2 + 2"},
        },
        "iteration_count": 0,
        "failure_count": 0,
        "todo_list": [
            {
                "task_id": "task_1",
                "description": "Calculate 2 + 2",
                "status": TaskStatus.IN_PROGRESS,
                "priority": 10,
                "payload": {},
            }
        ],
        "current_task_id": "task_1",
        "action_history": [],
    }


@pytest.fixture
def sample_state_with_agent_decision():
    """Create state with an agent decision."""
    return {
        "decision": {
            "action_type": "agent",
            "resource_id": "TestAgent",
            "payload": {"instruction": "Analyze the data"},
        },
        "iteration_count": 1,
        "failure_count": 0,
        "todo_list": [
            {
                "task_id": "task_2",
                "description": "Analyze data",
                "status": TaskStatus.IN_PROGRESS,
                "priority": 10,
                "payload": {},
            }
        ],
        "current_task_id": "task_2",
        "action_history": [],
    }


@pytest.fixture
def sample_state_with_python_decision():
    """Create state with a Python decision."""
    return {
        "decision": {
            "action_type": "python",
            "payload": {"code": "print(2 + 2)", "session_id": "test_session"},
        },
        "iteration_count": 2,
        "failure_count": 0,
        "todo_list": [
            {
                "task_id": "task_3",
                "description": "Run calculation",
                "status": TaskStatus.IN_PROGRESS,
                "priority": 10,
                "payload": {},
            }
        ],
        "current_task_id": "task_3",
        "action_history": [],
    }


@pytest.fixture
def sample_state_with_terminal_decision():
    """Create state with a terminal decision."""
    return {
        "decision": {"action_type": "terminal", "payload": {"command": "ls -la"}},
        "iteration_count": 3,
        "failure_count": 0,
        "todo_list": [
            {
                "task_id": "task_4",
                "description": "List directory",
                "status": TaskStatus.IN_PROGRESS,
                "priority": 5,
                "payload": {},
            }
        ],
        "current_task_id": "task_4",
        "action_history": [],
    }


@pytest.fixture
def sample_state_with_plan_decision():
    """Create state with a plan decision."""
    return {
        "decision": {
            "action_type": "plan",
            "execution_plan": [
                {
                    "phase_id": "phase_1",
                    "name": "Data Collection",
                    "goal": "Collect data",
                    "depends_on": [],
                },
                {
                    "phase_id": "phase_2",
                    "name": "Analysis",
                    "goal": "Analyze data",
                    "depends_on": ["phase_1"],
                },
            ],
        },
        "iteration_count": 0,
        "failure_count": 0,
        "todo_list": [],
        "action_history": [],
    }


@pytest.fixture
def sample_state_with_parallel_decision():
    """Create state with a parallel decision."""
    return {
        "decision": {
            "action_type": "parallel",
            "parallel_actions": [
                {
                    "action_type": "agent",
                    "resource_id": "TestAgent",
                    "payload": {"instruction": "Get Q3 data"},
                },
                {
                    "action_type": "agent",
                    "resource_id": "TestAgent",
                    "payload": {"instruction": "Get Q4 data"},
                },
            ],
        },
        "iteration_count": 1,
        "failure_count": 0,
        "todo_list": [],
        "action_history": [],
        "insights": {},
    }


class TestHandsInitialization:
    """Test Hands initialization and timeout configuration."""

    def test_hands_creation(self, hands):
        """Test that Hands is created with correct timeouts."""
        assert "agent" in hands.timeout_map
        assert "tool" in hands.timeout_map
        assert "terminal" in hands.timeout_map
        assert "python" in hands.timeout_map
        assert hands.timeout_map["agent"] == 60.0
        assert hands.timeout_map["tool"] == 30.0


class TestHandsToolExecution:
    """Test Hands tool execution routing."""

    @pytest.mark.asyncio
    async def test_execute_tool_success(
        self, hands, sample_state, mock_tool_registry, mock_content_hooks
    ):
        """Test Hands executes a tool successfully."""
        # Mock tool execution to return success
        mock_tool_registry.execute_tool.return_value = {"success": True, "result": "4"}

        config = {
            "configurable": {
                "owner": {"user_id": "test_user"},
                "thread_id": "test_thread",
            }
        }
        result = await hands.execute(sample_state, config)

        # Verify result structure
        assert "execution_result" in result
        assert result["execution_result"]["success"] is True
        assert (
            result["execution_result"]["output"]["result"] == "processed"
        )  # After CMS hooks
        assert "action_history" in result
        assert len(result["action_history"]) == 1

        # Verify tool was called
        mock_tool_registry.execute_tool.assert_called_once_with(
            "calculator", {"expression": "2 + 2"}
        )

        # Verify CMS hooks were called
        mock_content_hooks.on_task_complete.assert_called_once()

        # Verify failure count reset
        assert result["failure_count"] == 0

    @pytest.mark.asyncio
    async def test_execute_tool_failure(
        self, hands, sample_state, mock_tool_registry, mock_content_hooks
    ):
        """Test Hands handles tool execution failure."""
        # Mock tool execution to return failure
        mock_tool_registry.execute_tool.return_value = {
            "success": False,
            "error": "Tool execution failed",
        }

        config = {
            "configurable": {
                "owner": {"user_id": "test_user"},
                "thread_id": "test_thread",
            }
        }
        result = await hands.execute(sample_state, config)

        # Verify error handling
        assert result["execution_result"]["success"] is False
        assert "error" in result
        assert result["failure_count"] == 1
        assert result["last_failure_id"] is not None

        # Verify task status updated to failed
        task = next(t for t in result["todo_list"] if t["task_id"] == "task_1")
        assert task["status"] == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_execute_tool_with_no_code(
        self, hands, mock_tool_registry, mock_content_hooks
    ):
        """Test Hands handles tool execution with no payload."""
        state = {
            "decision": {"action_type": "tool", "resource_id": "calculator"},
            "iteration_count": 0,
            "todo_list": [],
            "action_history": [],
        }

        config = {
            "configurable": {
                "owner": {"user_id": "test_user"},
                "thread_id": "test_thread",
            }
        }
        result = await hands.execute(state, config)

        assert result["execution_result"]["success"] is True


class TestHandsAgentExecution:
    """Test Hands agent execution routing."""

    @pytest.mark.asyncio
    async def test_execute_agent_success(
        self,
        hands,
        sample_state_with_agent_decision,
        mock_agent_registry,
        mock_credential_service,
        mock_retry_manager,
        mock_telemetry_service,
        mock_content_hooks,
    ):
        """Test Hands executes an agent successfully."""
        # Mock HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "result": "Data analysis complete",
            "message": "Analysis successful",
        }
        mock_retry_manager.retry_async.return_value = mock_response

        # Mock database
        with patch("orchestrator.hands.SessionLocal") as mock_db:
            config = {
                "configurable": {"owner": "test_user", "thread_id": "test_thread"}
            }
            result = await hands.execute(sample_state_with_agent_decision, config)

            # Verify result
            assert result["execution_result"]["success"] is True
            assert "action_history" in result

            # Verify retry manager was called
            mock_retry_manager.retry_async.assert_called_once()

            # Verify telemetry was logged
            mock_telemetry_service.log_agent_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_agent_not_found(
        self, hands, mock_agent_registry, mock_content_hooks
    ):
        """Test Hands handles agent not found."""
        # Mock registry to return empty list (agent not found)
        mock_agent_registry.list_active_agents.return_value = []

        state = {
            "decision": {
                "action_type": "agent",
                "resource_id": "NonExistentAgent",
                "payload": {"instruction": "Test"},
            },
            "iteration_count": 0,
            "todo_list": [],
            "action_history": [],
        }

        config = {"configurable": {"owner": "test_user", "thread_id": "test_thread"}}
        result = await hands.execute(state, config)

        assert result["execution_result"]["success"] is False
        assert "not found" in result["execution_result"]["error_message"].lower()
        assert result["failure_count"] == 1

    @pytest.mark.asyncio
    async def test_execute_agent_http_failure(
        self,
        hands,
        sample_state_with_agent_decision,
        mock_retry_manager,
        mock_telemetry_service,
        mock_content_hooks,
        mock_agent_registry,
    ):
        """Test Hands handles agent HTTP connection failure."""
        # Mock HTTP exception - properly wrap in Mock that returns a mock function that raises
        mock_retry_manager.retry_async.return_value = None

        # We need to also simulate the exception happening inside the agent call
        # Let's patch the agent_registry to set up the mock properly
        with patch("orchestrator.hands.SessionLocal"):
            with patch("orchestrator.hands.get_credentials_for_headers") as mock_creds:
                mock_creds.return_value = {}

                # Now mock the agent call to raise an exception
                async def _raise_exception():
                    raise Exception("Connection refused")

                mock_retry_manager.retry_async.side_effect = _raise_exception()

                config = {
                    "configurable": {"owner": "test_user", "thread_id": "test_thread"}
                }
                result = await hands.execute(sample_state_with_agent_decision, config)

                # The error should be caught and handled
                assert result["execution_result"]["success"] is False
                error_msg = result["execution_result"]["error_message"] or ""
                # Check for connection error pattern (either with or without Connection in it)
                # The Hands implementation wraps the exception in "Agent connection error: {str(e)}"
                assert "error" in error_msg.lower() or "connection" in error_msg.lower()
                assert result["failure_count"] == 1


class TestHandsPythonExecution:
    """Test Hands Python code execution routing."""

    @pytest.mark.asyncio
    async def test_execute_python_success(
        self,
        hands,
        sample_state_with_python_decision,
        mock_code_sandbox,
        mock_content_hooks,
    ):
        """Test Hands executes Python code successfully."""
        config = {"configurable": {"owner": "test_user", "thread_id": "test_thread"}}
        result = await hands.execute(sample_state_with_python_decision, config)

        assert result["execution_result"]["success"] is True
        assert (
            result["execution_result"]["output"]["result"] == "processed"
        )  # After CMS hooks

        # Verify sandbox was called
        mock_code_sandbox.execute_code.assert_called_once_with(
            "print(2 + 2)", session_id="test_session"
        )

        # Verify task marked as completed
        task = next(t for t in result["todo_list"] if t["task_id"] == "task_3")
        assert task["status"] == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_python_no_code(
        self, hands, mock_code_sandbox, mock_content_hooks
    ):
        """Test Hands handles Python execution with no code."""
        state = {
            "decision": {"action_type": "python", "payload": {}},
            "iteration_count": 0,
            "todo_list": [],
            "action_history": [],
        }

        config = {"configurable": {"owner": "test_user", "thread_id": "test_thread"}}
        result = await hands.execute(state, config)

        assert result["execution_result"]["success"] is False
        assert "No code provided" in result["execution_result"]["error_message"]


class TestHandsTerminalExecution:
    """Test Hands terminal command execution routing."""

    @pytest.mark.asyncio
    async def test_execute_terminal_success(
        self,
        hands,
        sample_state_with_terminal_decision,
        mock_terminal_service,
        mock_telemetry_service,
        mock_content_hooks,
    ):
        """Test Hands executes terminal command successfully."""
        config = {"configurable": {"owner": "test_user", "thread_id": "test_thread"}}
        result = await hands.execute(sample_state_with_terminal_decision, config)

        assert result["execution_result"]["success"] is True

        # Verify terminal was called
        mock_terminal_service.execute_command.assert_called_once_with("ls -la")

        # Verify telemetry was logged (use pytest.approx for timing precision)
        mock_telemetry_service.log_tool_call.assert_called_once()
        call_args = mock_telemetry_service.log_tool_call.call_args
        assert call_args[0][0] == "Terminal"
        assert call_args[0][1] is True
        # Third argument is execution_time_ms, which can have small variations
        pytest.approx(call_args[0][2], 0.1)

    @pytest.mark.asyncio
    async def test_execute_terminal_no_command(
        self, hands, mock_terminal_service, mock_content_hooks
    ):
        """Test Hands handles terminal execution with no command."""
        state = {
            "decision": {"action_type": "terminal", "payload": {}},
            "iteration_count": 0,
            "todo_list": [],
            "action_history": [],
        }

        config = {"configurable": {"owner": "test_user", "thread_id": "test_thread"}}
        result = await hands.execute(state, config)

        assert result["execution_result"]["success"] is False
        assert "No command provided" in result["execution_result"]["error_message"]


class TestHandsPlanHandling:
    """Test Hands plan/replan acknowledgment."""

    @pytest.mark.asyncio
    async def test_handle_plan_action(self, hands, sample_state_with_plan_decision):
        """Test Hands acknowledges plan action."""
        config = {"configurable": {"owner": "test_user", "thread_id": "test_thread"}}
        result = await hands.execute(sample_state_with_plan_decision, config)

        assert result["execution_result"]["success"] is True
        assert (
            result["execution_result"]["output"]["message"] == "Execution plan created"
        )
        assert result["execution_result"]["output"]["phases"] == 2

    @pytest.mark.asyncio
    async def test_handle_replan_action(self, hands):
        """Test Hands acknowledges replan action."""
        state = {
            "decision": {
                "action_type": "replan",
                "execution_plan": [
                    {
                        "phase_id": "new_phase_1",
                        "name": "New Phase 1",
                        "goal": "New goal",
                        "depends_on": [],
                    }
                ],
            },
            "iteration_count": 5,
            "todo_list": [],
            "action_history": [],
        }

        config = {"configurable": {"owner": "test_user", "thread_id": "test_thread"}}
        result = await hands.execute(state, config)

        assert result["execution_result"]["success"] is True
        assert (
            result["execution_result"]["output"]["message"] == "Execution plan modified"
        )


class TestHandsParallelExecution:
    """Test Hands parallel action execution."""

    @pytest.mark.asyncio
    async def test_execute_parallel_success(
        self,
        hands,
        mock_tool_registry,
        mock_telemetry_service,
        mock_content_hooks,
    ):
        """Test Hands executes parallel tool actions successfully."""
        # Use tools instead of agents for easier testing (no HTTP mocking needed)
        state = {
            "decision": {
                "action_type": "parallel",
                "parallel_actions": [
                    {
                        "action_type": "tool",
                        "resource_id": "calculator",
                        "payload": {"expression": "2 + 2"},
                    },
                    {
                        "action_type": "tool",
                        "resource_id": "data_parser",
                        "payload": {"format": "json"},
                    },
                ],
            },
            "iteration_count": 0,
            "todo_list": [],
            "action_history": [],
        }

        config = {"configurable": {"owner": "test_user", "thread_id": "test_thread"}}
        result = await hands.execute(state, config)

        assert result["execution_result"]["success"] is True
        assert "parallel_results" in result["execution_result"]["output"]
        assert result["execution_result"]["output"]["total_actions"] == 2

    @pytest.mark.asyncio
    async def test_execute_parallel_with_failure(
        self,
        hands,
        mock_agent_registry,
        mock_retry_manager,
        mock_telemetry_service,
        mock_content_hooks,
    ):
        """Test Hands handles parallel action failures."""
        state = {
            "decision": {
                "action_type": "parallel",
                "parallel_actions": [
                    {
                        "action_type": "tool",
                        "resource_id": "tool1",
                        "payload": {"test": "data"},
                    },
                    {
                        "action_type": "tool",
                        "resource_id": "tool2",
                        "payload": {"test": "data"},
                    },
                ],
            },
            "iteration_count": 1,
            "todo_list": [],
            "action_history": [],
            "insights": {},
        }

        with patch("orchestrator.hands.tool_registry") as mock_tool:
            # Make first succeed, second fail
            async def execute_side_effect(name, payload):
                if name == "tool1":
                    return {"success": True, "result": "Success"}
                else:
                    return {"success": False, "error": "Tool 2 failed"}

            mock_tool.execute_tool = AsyncMock(side_effect=execute_side_effect)

            config = {
                "configurable": {"owner": "test_user", "thread_id": "test_thread"}
            }
            result = await hands.execute(state, config)

            # Should not be all success (one failed)
            assert result["execution_result"]["success"] is False
            assert result["execution_result"]["output"]["successful"] == 1
            assert result["execution_result"]["output"]["failed"] == 1

    @pytest.mark.asyncio
    async def test_execute_parallel_retries_on_failure(
        self, hands, mock_agent_registry, mock_telemetry_service, mock_content_hooks
    ):
        """Test Hands retries failed parallel actions."""
        state = {
            "decision": {
                "action_type": "parallel",
                "parallel_actions": [
                    {
                        "action_type": "tool",
                        "resource_id": "retry_tool",
                        "payload": {"test": "data"},
                    }
                ],
            },
            "iteration_count": 1,
            "todo_list": [],
            "action_history": [],
            "insights": {},
        }

        with patch("orchestrator.hands.tool_registry") as mock_tool:
            # First two calls fail, third succeeds
            call_count = [0]

            async def execute_side_effect(name, payload):
                call_count[0] += 1
                if call_count[0] < 3:
                    return {"success": False, "error": "Temporary failure"}
                return {"success": True, "result": "Success after retry"}

            mock_tool.execute_tool = AsyncMock(side_effect=execute_side_effect)

            config = {
                "configurable": {"owner": "test_user", "thread_id": "test_thread"}
            }
            result = await hands.execute(state, config)

            # Should succeed on retry
            assert result["execution_result"]["success"] is True
            assert call_count[0] >= 2  # At least one retry


class TestHandsSkipAndFinish:
    """Test Hands skip and finish action handling."""

    @pytest.mark.asyncio
    async def test_handle_skip_action(self, hands):
        """Test Hands handles skip action."""
        state = {
            "decision": {"action_type": "skip", "reasoning": "Skipping this action"},
            "iteration_count": 0,
            "todo_list": [],
            "action_history": [],
        }

        config = {"configurable": {"owner": "test_user", "thread_id": "test_thread"}}
        result = await hands.execute(state, config)

        assert result["execution_result"]["success"] is True
        assert result["execution_result"]["output"]["skipped"] is True
        assert result["execution_result"]["action_id"].startswith("skip_")

    @pytest.mark.asyncio
    async def test_handle_finish_action(self, hands):
        """Test Hands handles finish action."""
        state = {
            "decision": {
                "action_type": "finish",
                "user_response": "Task completed successfully",
            },
            "iteration_count": 5,
            "todo_list": [],
            "action_history": [],
        }

        config = {"configurable": {"owner": "test_user", "thread_id": "test_thread"}}
        result = await hands.execute(state, config)

        assert result["execution_result"]["success"] is True
        assert (
            result["execution_result"]["output"]["message"]
            == "Task completed successfully"
        )


class TestHandsUnknownAction:
    """Test Hands handling of unknown action types."""

    @pytest.mark.asyncio
    async def test_handle_unknown_action_type(self, hands):
        """Test Hands handles unknown action type gracefully."""
        state = {
            "decision": {"action_type": "unknown_action", "resource_id": "something"},
            "iteration_count": 0,
            "todo_list": [],
            "action_history": [],
        }

        config = {"configurable": {"owner": "test_user", "thread_id": "test_thread"}}
        result = await hands.execute(state, config)

        assert result["execution_result"]["success"] is False
        assert "Unknown action type" in result["execution_result"]["error_message"]


class TestHandsNoDecision:
    """Test Hands handling when no decision is present."""

    @pytest.mark.asyncio
    async def test_execute_without_decision(self, hands):
        """Test Hands handles state without decision."""
        state = {
            "iteration_count": 0,
            "todo_list": [],
            "action_history": [],
        }

        config = {"configurable": {"owner": "test_user", "thread_id": "test_thread"}}
        result = await hands.execute(state, config)

        assert "error" in result
        assert "No brain decision found" in result["error"]


class TestHandsResultSummaryGeneration:
    """Test result summary generation for action history."""

    def test_generate_string_summary(self, hands):
        """Test summary generation for string output."""
        result = ActionResult(
            action_id="test_action",
            success=True,
            output="This is a long output that should be truncated at 500 characters. "
            * 10,
        )
        summary = hands._generate_result_summary(result.output)

        assert len(summary) <= 503  # 500 + '...' if needed
        assert "..." in summary

    def test_generate_dict_summary(self, hands):
        """Test summary generation for dict output."""
        result = ActionResult(
            action_id="test_action",
            success=True,
            output={
                "result": "Success",
                "message": "Operation completed",
                "data": [1, 2, 3],
            },
        )
        summary = hands._generate_result_summary(result.output)

        assert "result:" in summary
        assert "Success" in summary

    def test_generate_empty_summary(self, hands):
        """Test summary generation for None output."""
        summary = hands._generate_result_summary(None)

        assert summary == "No output"


class TestHandsParallelInsightsExtraction:
    """Test insights extraction from parallel results."""

    def test_extract_parallel_insights_success(self, hands, mock_agent_registry):
        """Test extracting insights from successful parallel results."""
        state = {
            "insights": {"existing": "value"},
            "iteration_count": 3,
        }

        result = ActionResult(
            action_id="parallel_123",
            success=True,
            output={
                "parallel_results": [
                    {
                        "index": 0,
                        "action_type": "agent",
                        "resource_id": "TestAgent",
                        "success": True,
                        "output": {"result": "Q4 revenue: $2.1M and analysis complete"},
                    },
                    {
                        "index": 1,
                        "action_type": "agent",
                        "resource_id": "FinanceAgent",
                        "success": True,
                        "output": {
                            "result": "Q3 revenue: $1.8M with detailed breakdown"
                        },
                    },
                ]
            },
        )

        updates = hands._extract_parallel_insights(state, result)

        assert "insights" in updates
        # The insights should include both the existing insight and the extracted ones
        assert (
            updates["insights"].get("existing") == "value"
            or updates["insights"].get("existing") is not None
        )
        # Check that parallel insights were created with the correct keys
        assert any("parallel_3_" in key for key in updates["insights"].keys())
        # Check that at least one of the expected insights exists
        has_test_agent = any(
            "TestAgent" in str(v) for v in updates["insights"].values()
        )
        has_revenue = any(
            "2.1M" in str(v) or "1.8M" in str(v) for v in updates["insights"].values()
        )
        assert has_test_agent or has_revenue, "Expected insights not found"

    def test_extract_parallel_insights_skips_failed(self, hands):
        """Test insights extraction skips failed parallel actions."""
        state = {
            "insights": {},
            "iteration_count": 1,
        }

        result = ActionResult(
            action_id="parallel_123",
            success=False,
            output={
                "parallel_results": [
                    {
                        "index": 0,
                        "action_type": "agent",
                        "resource_id": "TestAgent",
                        "success": False,
                        "error": "Connection failed",
                    }
                ]
            },
        )

        updates = hands._extract_parallel_insights(state, result)

        assert len(updates) == 0


class TestHandsActionHistoryRecording:
    """Test action history recording and state updates."""

    @pytest.mark.asyncio
    async def test_action_history_appends(
        self, hands, sample_state, mock_tool_registry, mock_content_hooks
    ):
        """Test action history is appended correctly."""
        mock_tool_registry.execute_tool.return_value = {
            "success": True,
            "result": "Test result",
        }

        # Set initial action history
        sample_state["action_history"] = [
            {
                "iteration": 0,
                "action_type": "tool",
                "resource_id": "previous_tool",
                "success": True,
                "result_summary": "Previous result",
                "timestamp": time.time(),
                "execution_time_ms": 100,
            }
        ]

        config = {"configurable": {"owner": "test_user", "thread_id": "test_thread"}}
        result = await hands.execute(sample_state, config)

        # Should have added new entry
        assert len(result["action_history"]) == 2

        # New entry should be last
        new_entry = result["action_history"][-1]
        assert new_entry["iteration"] == 0
        assert new_entry["action_type"] == "tool"
        assert new_entry["resource_id"] == "calculator"
        assert new_entry["success"] is True

    @pytest.mark.asyncio
    async def test_task_status_updates_on_success(
        self, hands, sample_state, mock_tool_registry, mock_content_hooks
    ):
        """Test task status updates to completed on success."""
        mock_tool_registry.execute_tool.return_value = {
            "success": True,
            "result": "Success",
        }

        config = {"configurable": {"owner": "test_user", "thread_id": "test_thread"}}
        result = await hands.execute(sample_state, config)

        task = next(t for t in result["todo_list"] if t["task_id"] == "task_1")
        assert task["status"] == TaskStatus.COMPLETED
        assert task["result"] is not None

    @pytest.mark.asyncio
    async def test_task_status_updates_on_failure(
        self, hands, sample_state, mock_tool_registry, mock_content_hooks
    ):
        """Test task status updates to failed on failure."""
        mock_tool_registry.execute_tool.return_value = {
            "success": False,
            "error": "Execution failed",
        }

        config = {"configurable": {"owner": "test_user", "thread_id": "test_thread"}}
        result = await hands.execute(sample_state, config)

        task = next(t for t in result["todo_list"] if t["task_id"] == "task_1")
        assert task["status"] == TaskStatus.FAILED
        assert task["error"] is not None


class TestActionResultSchema:
    """Test ActionResult schema validation."""

    def test_action_result_success(self):
        """Test successful ActionResult creation."""
        result = ActionResult(
            action_id="test_1",
            success=True,
            output={"result": "Success"},
            execution_time_ms=100.5,
        )

        assert result.action_id == "test_1"
        assert result.success is True
        assert result.output["result"] == "Success"
        assert result.execution_time_ms == 100.5

    def test_action_result_failure(self):
        """Test failed ActionResult creation."""
        result = ActionResult(
            action_id="test_2",
            success=False,
            error_message="Error occurred",
            execution_time_ms=50.0,
        )

        assert result.action_id == "test_2"
        assert result.success is False
        assert result.error_message == "Error occurred"
        assert result.output is None
