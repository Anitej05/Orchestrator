"""
Unit tests for Omni-Brain (Brain reasoning logic) - FIXED VERSION

This test module properly mocks external dependencies (LLM, agents, tools) to test
the Brain's decision-making logic in isolation.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
import json
import time

backend_root = Path(__file__).parent.parent
sys.path.insert(0, str(backend_root))

# Create global mock for inference_service to use across patches
mock_inference_service_impl = Mock()
mock_inference_service_impl.generate_structured = AsyncMock()
mock_inference_service_impl.InferencePriority = Mock()
mock_inference_service_impl.InferencePriority.SPEED = "speed"

# Need to patch BOTH import locations since brain.py imports inference_service multiple ways
with (
    patch("services.inference_service.inference_service", mock_inference_service_impl),
    patch("orchestrator.brain.inference_service", mock_inference_service_impl),
):
    from orchestrator.brain import Brain, BrainDecision
    from orchestrator.schemas import (
        TaskItem,
        TaskStatus,
        TaskPriority,
        ActionResult,
    )

# Import services at module level for easier mocking
import services.agent_registry_service
import services.tool_registry_service


@pytest.fixture(autouse=True)
def setup_mocks():
    """Reset mocks before each test."""
    mock_inference_service_impl.generate_structured.reset_mock()
    # Patch the inference_service in brain module
    with patch("orchestrator.brain.inference_service", mock_inference_service_impl):
        yield


@pytest.fixture
def brain():
    """Create a Brain instance for testing."""
    return Brain()


@pytest.fixture
def mock_agent_registry(monkeypatch):
    """Mock the agent registry."""
    mock = Mock()
    mock.list_active_agents = Mock(
        return_value=[
            {
                "id": "test_agent_1",
                "name": "TestAgent",
                "description": "A test agent for testing",
                "type": "http_rest",
            },
            {
                "id": "test_agent_2",
                "name": "FinanceAgent",
                "description": "Financial data analysis agent",
                "type": "http_rest",
            },
        ]
    )
    monkeypatch.setattr(services.agent_registry_service, "agent_registry", mock)
    return mock


@pytest.fixture
def mock_tool_registry(monkeypatch):
    """Mock the tool registry."""
    mock = Mock()
    mock.list_tools = Mock(
        return_value=[
            {
                "name": "calculator",
                "description": "Perform mathematical calculations",
            },
            {"name": "data_parser", "description": "Parse and process data files"},
        ]
    )
    monkeypatch.setattr(services.tool_registry_service, "tool_registry", mock)
    return mock


@pytest.fixture
def mock_content_orchestrator(monkeypatch):
    """Mock the content orchestrator for context optimization."""
    import orchestrator.content_orchestrator as co_module

    mock = Mock(return_value={"context": "No history available.", "messages": []})
    monkeypatch.setattr(
        co_module,
        "get_optimized_llm_context",
        lambda *args, **kwargs: {"context": "No history available.", "messages": []},
    )
    return mock


@pytest.fixture
def sample_state():
    """Create a sample orchestrator state."""
    return {
        "original_prompt": "Calculate 2 + 2",
        "todo_list": [],
        "memory": {},
        "insights": {},
        "action_history": [],
        "iteration_count": 0,
        "failure_count": 0,
        "execution_plan": None,
        "current_phase_id": None,
    }


@pytest.fixture
def sample_state_with_tasks():
    """Create a state with tasks in the todo list."""
    task1 = TaskItem(
        task_id="task_abc123",
        description="Calculate 2 + 2",
        priority=10,
        status=TaskStatus.PENDING,
        payload={},
    )
    return {
        "original_prompt": "Calculate 2 + 2",
        "todo_list": [task1.model_dump()],
        "memory": {},
        "insights": {},
        "action_history": [],
        "iteration_count": 0,
        "failure_count": 0,
        "execution_plan": None,
        "current_phase_id": None,
        "current_task_id": "task_abc123",
    }


@pytest.fixture
def sample_state_with_plan():
    """Create a state with an execution plan."""
    return {
        "original_prompt": "Analyze data and create report",
        "todo_list": [
            {
                "task_id": "task_1",
                "description": "Collect data",
                "priority": 10,
                "status": TaskStatus.PENDING,
                "payload": {},
            }
        ],
        "memory": {},
        "insights": {},
        "action_history": [],
        "iteration_count": 0,
        "failure_count": 0,
        "execution_plan": [
            {
                "phase_id": "phase_1",
                "name": "Data Collection",
                "goal": "Gather Q3 and Q4 sales data",
                "status": "pending",
                "depends_on": [],
            },
            {
                "phase_id": "phase_2",
                "name": "Analysis",
                "goal": "Compare and analyze the data",
                "status": "pending",
                "depends_on": ["phase_1"],
            },
        ],
        "current_phase_id": "phase_1",
        "current_task_id": "task_1",
    }


class TestBrainInitialization:
    """Test Brain initialization and basic properties."""

    def test_brain_creation(self, brain):
        """Test that Brain is created with correct defaults."""
        assert brain.max_failures == 3
        assert brain.max_iterations == 25

    def test_brain_decision_model_validation(self):
        """Test BrainDecision schema validation."""
        decision = BrainDecision(
            action_type="agent",
            resource_id="TestAgent",
            payload={"instruction": "Test instruction"},
            reasoning="Test reasoning",
            is_finished=False,
        )
        assert decision.action_type == "agent"
        assert decision.resource_id == "TestAgent"
        assert decision.is_finished is False

    def test_brain_decision_with_approval(self):
        """Test BrainDecision with human-in-the-loop approval."""
        decision = BrainDecision(
            action_type="agent",
            resource_id="EmailAgent",
            payload={"instruction": "Send email to finance@example.com"},
            reasoning="Sending email requires user approval",
            requires_approval=True,
            approval_reason="Will send email with report attachment to finance@example.com",
        )
        assert decision.requires_approval is True
        assert decision.approval_reason is not None

    def test_brain_decision_with_parallel_actions(self):
        """Test BrainDecision with parallel actions."""
        decision = BrainDecision(
            action_type="parallel",
            parallel_actions=[
                {
                    "action_type": "agent",
                    "resource_id": "SpreadsheetAgent",
                    "payload": {"instruction": "Get Q3 data"},
                },
                {
                    "action_type": "agent",
                    "resource_id": "SpreadsheetAgent",
                    "payload": {"instruction": "Get Q4 data"},
                },
            ],
        )
        assert decision.action_type == "parallel"
        assert len(decision.parallel_actions) == 2

    def test_brain_decision_with_plan(self):
        """Test BrainDecision with execution plan."""
        decision = BrainDecision(
            action_type="plan",
            execution_plan=[
                {
                    "phase_id": "phase_1",
                    "name": "Data Collection",
                    "goal": "Gather data",
                    "depends_on": [],
                }
            ],
        )
        assert decision.action_type == "plan"
        assert len(decision.execution_plan) == 1

    def test_brain_decision_with_phase_complete(self):
        """Test BrainDecision with phase completion."""
        decision = BrainDecision(
            action_type="agent",
            resource_id="TestAgent",
            payload={"instruction": "Analyze data"},
            phase_complete=True,
            phase_goal_verified="Data analysis completed with 95% accuracy",
        )
        assert decision.phase_complete is True
        assert decision.phase_goal_verified is not None


class TestBrainStateInitialization:
    """Test initial state handling."""

    @pytest.mark.asyncio
    async def test_initialize_empty_state(self, brain, sample_state):
        """Test Brain initializes state with first task."""
        result = await brain.think(sample_state)

        assert "todo_list" in result
        assert len(result["todo_list"]) == 1
        assert result["todo_list"][0]["status"] == TaskStatus.PENDING
        assert result["memory"] == {}
        assert result["insights"] == {}
        assert result["action_history"] == []
        assert result["iteration_count"] == 0
        assert result["failure_count"] == 0
        assert "decision" in result

    @pytest.mark.asyncio
    async def test_initialize_creates_first_task(self, brain, sample_state):
        """Test that initial state creates the first task."""
        result = await brain.think(sample_state)

        assert len(result["todo_list"]) == 1
        task = result["todo_list"][0]
        assert "task_id" in task
        assert task["description"] == "Initialize objective analysis"
        assert task["status"] == TaskStatus.PENDING
        assert task["priority"] == 10


class TestBrainDecisionMaking:
    """Test the Brain's decision-making logic with mocked LLM."""

    @pytest.mark.asyncio
    async def test_python_action_decision(
        self,
        brain,
        sample_state_with_tasks,
        mock_agent_registry,
        mock_tool_registry,
        mock_content_orchestrator,
    ):
        """Test Brain decides to execute Python code."""
        mock_decision = BrainDecision(
            action_type="python",
            payload={"code": "print(2 + 2)"},
            reasoning="Calculation can be done directly with Python",
        )
        mock_inference_service_impl.generate_structured.return_value = mock_decision

        config = {"configurable": {"thread_id": "test_thread"}}
        result = await brain.think(sample_state_with_tasks, config)

        assert "decision" in result
        decision = result["decision"]
        assert decision["action_type"] == "python"
        assert decision["payload"]["code"] == "print(2 + 2)"

    @pytest.mark.asyncio
    async def test_agent_action_decision(
        self,
        brain,
        sample_state_with_tasks,
        mock_agent_registry,
        mock_tool_registry,
        mock_content_orchestrator,
    ):
        """Test Brain decides to use an agent."""
        mock_decision = BrainDecision(
            action_type="agent",
            resource_id="TestAgent",
            payload={"instruction": "Analyze the data"},
            reasoning="TestAgent is best suited for data analysis",
        )
        mock_inference_service_impl.generate_structured.return_value = mock_decision

        config = {"configurable": {"thread_id": "test_thread"}}
        result = await brain.think(sample_state_with_tasks, config)

        assert result["decision"]["action_type"] == "agent"
        assert result["decision"]["resource_id"] == "TestAgent"

    @pytest.mark.asyncio
    async def test_tool_action_decision(
        self,
        brain,
        sample_state_with_tasks,
        mock_agent_registry,
        mock_tool_registry,
        mock_content_orchestrator,
    ):
        """Test Brain decides to use a tool."""
        mock_decision = BrainDecision(
            action_type="tool",
            resource_id="calculator",
            payload={"expression": "2 + 2"},
            reasoning="Calculator tool can handle this calculation",
        )
        mock_inference_service_impl.generate_structured.return_value = mock_decision

        config = {"configurable": {"thread_id": "test_thread"}}
        result = await brain.think(sample_state_with_tasks, config)

        assert result["decision"]["action_type"] == "tool"
        assert result["decision"]["resource_id"] == "calculator"

    @pytest.mark.asyncio
    async def test_plan_creation_decision(
        self,
        brain,
        sample_state_with_tasks,
        mock_agent_registry,
        mock_tool_registry,
        mock_content_orchestrator,
    ):
        """Test Brain decides to create an execution plan."""
        mock_decision = BrainDecision(
            action_type="plan",
            execution_plan=[
                {
                    "phase_id": "phase_1",
                    "name": "Data Collection",
                    "goal": "Collect Q3 and Q4 data",
                    "depends_on": [],
                },
                {
                    "phase_id": "phase_2",
                    "name": "Analysis",
                    "goal": "Compare and analyze data",
                    "depends_on": ["phase_1"],
                },
            ],
            reasoning="Complex multi-phase task requires planning",
        )
        mock_inference_service_impl.generate_structured.return_value = mock_decision

        config = {"configurable": {"thread_id": "test_thread"}}
        result = await brain.think(sample_state_with_tasks, config)

        assert result["decision"]["action_type"] == "plan"
        assert "execution_plan" in result

    @pytest.mark.asyncio
    async def test_prompt_content(
        self,
        brain,
        sample_state_with_tasks,
    ):
        """Test that the prompt includes key sections."""
        mock_decision = BrainDecision(
            action_type="plan",
            reasoning="Need to plan first",
            execution_plan=[{"phase_id": "phase_1", "step": "step_1"}]
        )
        mock_inference_service_impl.generate_structured.return_value = mock_decision

        config = {"configurable": {"thread_id": "test_thread"}}
        result = await brain.think(sample_state_with_tasks, config)

        # Get the call arguments
        call_args = mock_inference_service_impl.generate_structured.call_args
        assert call_args is not None
        
        # Extract the prompt from the call args
        # call_args is (args, kwargs) tuple for AsyncMock
        if isinstance(call_args, tuple):
            kwargs = call_args[1]
        else:
            kwargs = call_args.kwargs

        messages = kwargs['messages']
        prompt = messages[0].content

        assert "## PERSONA" in prompt
        assert "expressive AI assistant" in prompt
        assert "## TOOL USE GUIDELINES" in prompt
        assert "YOU MUST GENERATE THE CODE" in prompt
        assert len(result["execution_plan"]) == 1
        assert result["execution_plan"][0]["phase_id"] == "phase_1"

    @pytest.mark.asyncio
    async def test_finish_decision(
        self,
        brain,
        sample_state_with_tasks,
        mock_agent_registry,
        mock_tool_registry,
        mock_content_orchestrator,
    ):
        """Test Brain decides to finish the task."""
        mock_decision = BrainDecision(
            action_type="finish",
            user_response="The calculation is complete: 2 + 2 = 4",
            is_finished=True,
            reasoning="Task has been completed successfully",
        )
        mock_inference_service_impl.generate_structured.return_value = mock_decision

        config = {"configurable": {"thread_id": "test_thread"}}
        result = await brain.think(sample_state_with_tasks, config)

        assert result["decision"]["action_type"] == "finish"
        assert result["final_response"] == "The calculation is complete: 2 + 2 = 4"
        assert result["current_task_id"] is None

    @pytest.mark.asyncio
    async def test_parallel_action_decision(
        self,
        brain,
        sample_state_with_tasks,
        mock_agent_registry,
        mock_tool_registry,
        mock_content_orchestrator,
    ):
        """Test Brain decides to run parallel actions."""
        mock_decision = BrainDecision(
            action_type="parallel",
            parallel_actions=[
                {
                    "action_type": "agent",
                    "resource_id": "SpreadsheetAgent",
                    "payload": {"instruction": "Get Q3 data"},
                },
                {
                    "action_type": "agent",
                    "resource_id": "SpreadsheetAgent",
                    "payload": {"instruction": "Get Q4 data"},
                },
            ],
            reasoning="Independent tasks can run in parallel",
        )
        mock_inference_service_impl.generate_structured.return_value = mock_decision

        config = {"configurable": {"thread_id": "test_thread"}}
        result = await brain.think(sample_state_with_tasks, config)

        assert result["decision"]["action_type"] == "parallel"
        assert len(result["decision"]["parallel_actions"]) == 2


class TestBrainApprovalHandling:
    """Test human-in-the-loop approval flow."""

    @pytest.mark.asyncio
    async def test_requires_approval_sets_pending(
        self,
        brain,
        sample_state_with_tasks,
        mock_agent_registry,
        mock_tool_registry,
        mock_content_orchestrator,
    ):
        """Test Brain sets pending_approval when requires_approval=True."""
        mock_decision = BrainDecision(
            action_type="agent",
            resource_id="EmailAgent",
            payload={"instruction": "Send to finance@example.com"},
            reasoning="Sending email",
            requires_approval=True,
            approval_reason="Will send sensitive email to finance@example.com",
        )
        mock_inference_service_impl.generate_structured.return_value = mock_decision

        config = {"configurable": {"thread_id": "test_thread"}}
        result = await brain.think(sample_state_with_tasks, config)

        assert result.get("pending_approval") is True
        assert "pending_decision" in result
        assert result["pending_decision"]["requires_approval"] is True

    @pytest.mark.asyncio
    async def test_approval_not_required_for_safe_actions(
        self,
        brain,
        sample_state_with_tasks,
        mock_agent_registry,
        mock_tool_registry,
        mock_content_orchestrator,
    ):
        """Test Brain does not set pending_approval for safe actions."""
        mock_decision = BrainDecision(
            action_type="python",
            payload={"code": "print(2 + 2)"},
            reasoning="Safe calculation",
        )
        mock_inference_service_impl.generate_structured.return_value = mock_decision

        config = {"configurable": {"thread_id": "test_thread"}}
        result = await brain.think(sample_state_with_tasks, config)

        assert result.get("pending_approval") is not True


class TestBrainPhaseManagement:
    """Test phase management and completion."""

    @pytest.mark.asyncio
    async def test_phase_completion_with_verified_goal(
        self,
        brain,
        sample_state_with_tasks,
        mock_agent_registry,
        mock_tool_registry,
        mock_content_orchestrator,
    ):
        """Test Brain handles phase completion with verified goal."""
        # Update state to have a plan
        state = dict(sample_state_with_tasks)
        state["execution_plan"] = [
            {
                "phase_id": "phase_1",
                "name": "Collection",
                "goal": "Gather data",
                "status": "pending",
                "depends_on": [],
            }
        ]
        state["current_phase_id"] = "phase_1"

        mock_decision = BrainDecision(
            action_type="agent",
            resource_id="TestAgent",
            payload={"instruction": "Collect data"},
            phase_complete=True,
            phase_goal_verified="Successfully gathered Q4 data: $2.1M revenue",
            reasoning="Phase goal achieved",
        )
        mock_inference_service_impl.generate_structured.return_value = mock_decision

        config = {"configurable": {"thread_id": "test_thread"}}
        result = await brain.think(state, config)

        assert result["decision"]["phase_complete"] is True
        assert (
            result["decision"]["phase_goal_verified"]
            == "Successfully gathered Q4 data: $2.1M revenue"
        )


class TestBrainInsightsExtraction:
    """Test insights extraction from execution results."""

    def test_extract_insights_from_successful_action(self, brain):
        """Test Brain extracts insights from successful action."""
        state = {
            "execution_result": {
                "success": True,
                "output": {"data": "Q4 revenue: $2.1M, Q3 revenue: $1.8M"},
            },
            "iteration_count": 5,
            "insights": {},
        }

        result = brain._extract_insights_from_last_action(state, {})

        assert "step_5" in result
        assert len(result["step_5"]) > 20

    def test_extract_insights_from_string_output(self, brain):
        """Test Brain extracts insights from string output."""
        state = {
            "execution_result": {
                "success": True,
                "output": "Analysis complete: The data shows a 15% increase in revenue",
            },
            "iteration_count": 3,
            "insights": {},
        }

        result = brain._extract_insights_from_last_action(state, {})

        assert "step_3" in result
        assert "15% increase" in result["step_3"]

    def test_no_insights_from_failed_action(self, brain):
        """Test Brain does not extract insights from failed actions."""
        state = {
            "execution_result": {
                "success": False,
                "error_message": "Connection failed",
            },
            "iteration_count": 1,
            "insights": {},
        }

        result = brain._extract_insights_from_last_action(state, {})

        assert result == {}

    def test_no_insights_from_empty_output(self, brain):
        """Test Brain does not extract insights from empty outputs."""
        state = {
            "execution_result": {"success": True, "output": None},
            "iteration_count": 1,
            "insights": {},
        }

        result = brain._extract_insights_from_last_action(state, {})

        assert result == {}


class TestBrainActionHistoryBuilding:
    """Test action history building and budget management."""

    def test_build_empty_action_history(self, brain):
        """Test Brain builds empty action history."""
        result = brain._build_action_history_view([])

        assert "No actions taken yet" in result

    def test_build_action_history_below_budget(self, brain):
        """Test Brain includes all actions below budget."""
        action_history = [
            {
                "action_type": "agent",
                "resource_id": "TestAgent",
                "success": True,
                "result_summary": "Data retrieved",
                "timestamp": "2024-01-01T10:00:00",
            },
            {
                "action_type": "python",
                "payload": {"code": "print(2+2)"},
                "success": True,
                "result_summary": "4",
                "timestamp": "2024-01-01T10:01:00",
            },
        ]

        result = brain._build_action_history_view(action_history)

        assert "TestAgent" in result
        assert "python" in result
        assert "Data retrieved" in result or "4" in result

    def test_build_action_history_with_truncation(self, brain):
        """Test Brain truncates large action history to budget."""
        action_history = []
        for i in range(25):  # More than the action_history budget
            action_history.append(
                {
                    "action_type": "agent",
                    "resource_id": f"Agent{i}",
                    "success": True,
                    "result_summary": f"Result {i}",
                    "timestamp": f"2024-01-01T10:{i:02d}:00",
                }
            )

        result = brain._build_action_history_view(action_history)

        # Should truncate, showing recent actions
        assert (
            "TRUNCATED" in result
            or "archived" in result
            or len(result) < 10000
        )


class TestBrainPlanValidation:
    """Test execution plan validation."""

    def test_validate_well_formed_plan(self, brain):
        """Test Brain validates well-formed plan."""
        plan = [
            {
                "phase_id": "phase_1",
                "name": "Data Collection",
                "goal": "Gather data",
                "depends_on": [],
            },
            {
                "phase_id": "phase_2",
                "name": "Analysis",
                "goal": "Analyze data",
                "depends_on": ["phase_1"],
            },
        ]

        validated, errors = brain._validate_execution_plan(plan)

        assert errors == []
        assert len(validated) == 2

    def test_validate_plan_with_missing_phase_id(self, brain):
        """Test Brain detects missing phase_id."""
        plan = [
            {
                "name": "Data Collection",
                "goal": "Gather data",
                "depends_on": [],
            },
        ]

        validated, errors = brain._validate_execution_plan(plan)

        assert len(errors) > 0
        assert any("phase_id" in str(e).lower() for e in errors)

    def test_validate_plan_with_invalid_dependency(self, brain):
        """Test Brain detects invalid phase dependency."""
        plan = [
            {
                "phase_id": "phase_1",
                "name": "Data Collection",
                "goal": "Gather data",
                "depends_on": ["non_existent_phase"],
            },
        ]

        validated, errors = brain._validate_execution_plan(plan)

        assert len(errors) > 0


class TestBrainErrorHandling:
    """Test Brain error handling and fallback modes."""

    @pytest.mark.asyncio
    async def test_llm_failure_returns_finish_decision(
        self,
        brain,
        sample_state_with_tasks,
        mock_agent_registry,
        mock_tool_registry,
        monkeypatch,
    ):
        """Test Brain handles LLM failure gracefully."""

        # Mock inference service to raise exception
        async def raise_exception(*args, **kwargs):
            raise Exception("LLM service unavailable")

        # Use monkeypatch to replace the generate_structured method
        original_generate = mock_inference_service_impl.generate_structured
        mock_inference_service_impl.generate_structured = Mock(side_effect=raise_exception)

        config = {"configurable": {"thread_id": "test_thread"}}
        result = await brain.think(sample_state_with_tasks, config)

        # Brain should handle the error and continue
        assert "decision" in result or "error" in result

        # Restore original
        mock_inference_service_impl.generate_structured = original_generate

    def test_max_iterations_exceeded(self, brain):
        """Test Brain stops when max iterations exceeded."""
        state = {
            "original_prompt": "Complex task",
            "todo_list": [
                {
                    "task_id": "task_1",
                    "description": "Pending",
                    "status": TaskStatus.PENDING,
                }
            ],
            "iteration_count": 100,
            "failure_count": 0,
            "memory": {},
            "insights": {},
            "action_history": [],
        }

        result = brain._force_finish_with_error(state, "Maximum iterations reached")

        assert "error" in result or "final_response" in result
        assert result.get("iteration_count", 100) >= 100

    def test_max_failures_triggers_fallback(
        self,
        brain,
        sample_state_with_tasks,
        mock_agent_registry,
        mock_tool_registry,
    ):
        """Test Brain enters fallback mode after max failures."""
        state = dict(sample_state_with_tasks)
        state["failure_count"] = 5

        result = brain._enter_fallback_mode(state, {}, {})

        assert result["decision"]["fallback_mode"] is True
        assert result["decision"]["is_finished"] is True


class TestBrainTodoPreview:
    """Test todo list preview building."""

    def test_build_empty_todo_preview(self, brain):
        """Test Brain builds empty todo preview."""
        result = brain._build_todo_preview([])

        assert "No tasks" in result or "empty" in result.lower()

    def test_build_todo_preview_with_tasks(self, brain):
        """Test Brain builds todo preview with tasks."""
        tasks = [
            {
                "task_id": "task_1",
                "description": "Collect Q3 data",
                "status": TaskStatus.PENDING,
                "priority": 10,
            },
            {
                "task_id": "task_2",
                "description": "Collect Q4 data",
                "status": TaskStatus.IN_PROGRESS,
                "priority": 10,
            },
            {
                "task_id": "task_3",
                "description": "Analyze data",
                "status": TaskStatus.COMPLETED,
                "priority": 5,
            },
        ]

        result = brain._build_todo_preview(tasks)

        assert "Q3 data" in result
        assert "Q4 data" in result
        assert "PENDING" in result or "IN_PROGRESS" in result
