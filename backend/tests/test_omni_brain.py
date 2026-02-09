"""
Unit tests for Omni-Brain (Brain reasoning logic).

This test module mocks external dependencies (LLM, agents, tools) to test
the Brain's decision-making logic in isolation.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
import json
import time

import sys
from pathlib import Path

backend_root = Path(__file__).parent.parent
sys.path.insert(0, str(backend_root))

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


@pytest.fixture
def mock_inference_service():
    """Mock the inference service for LLM calls."""
    with patch("orchestrator.brain.inference_service") as mock:
        mock.generate_structured = AsyncMock()
        mock.InferencePriority = Mock()
        mock.InferencePriority.SPEED = "speed"
        yield mock


@pytest.fixture
def mock_agent_registry():
    """Mock the agent registry."""
    # Patch the import in orchestrator.brain._make_decision
    with patch.object(services.agent_registry_service, "agent_registry") as mock:
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
        yield mock


@pytest.fixture
def mock_tool_registry():
    """Mock the tool registry."""
    # Patch the import in orchestrator.brain._make_decision
    with patch.object(services.tool_registry_service, "tool_registry") as mock:
        mock.list_tools = Mock(
            return_value=[
                {
                    "name": "calculator",
                    "description": "Perform mathematical calculations",
                },
                {"name": "data_parser", "description": "Parse and process data files"},
            ]
        )
        yield mock


@pytest.fixture
def mock_content_orchestrator():
    """Mock the content orchestrator for context optimization."""
    with patch("orchestrator.brain.get_optimized_llm_context") as mock:
        mock.return_value = {"context": "No history available.", "messages": []}
        yield mock


@pytest.fixture
def brain():
    """Create a Brain instance for testing."""
    return Brain()


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


class TestBrainStateInitialization:
    """Test initial state handling."""

    @pytest.mark.asyncio
    async def test_initialize_empty_state(
        self, brain, sample_state, mock_inference_service
    ):
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
    async def test_initialize_creates_first_task(
        self, brain, sample_state, mock_inference_service
    ):
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
        mock_inference_service,
        mock_agent_registry,
        mock_tool_registry,
        mock_content_orchestrator,
    ):
        """Test Brain decides to execute Python code."""
        # Mock LLM to return a Python action decision
        mock_decision = BrainDecision(
            action_type="python",
            payload={"code": "print(2 + 2)"},
            reasoning="Calculation can be done directly with Python",
        )
        mock_inference_service.generate_structured.return_value = mock_decision

        config = {"configurable": {"thread_id": "test_thread"}}
        result = await brain.think(sample_state_with_tasks, config)

        assert "decision" in result
        decision = result["decision"]
        assert decision["action_type"] == "python"
        assert decision["payload"]["code"] == "print(2 + 2)"

        # Verify LLM was called with correct parameters
        mock_inference_service.generate_structured.assert_called_once()
        call_args = mock_inference_service.generate_structured.call_args
        assert call_args is not None

    @pytest.mark.asyncio
    async def test_agent_action_decision(
        self,
        brain,
        sample_state_with_tasks,
        mock_inference_service,
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
        mock_inference_service.generate_structured.return_value = mock_decision

        config = {"configurable": {"thread_id": "test_thread"}}
        result = await brain.think(sample_state_with_tasks, config)

        assert result["decision"]["action_type"] == "agent"
        assert result["decision"]["resource_id"] == "TestAgent"

    @pytest.mark.asyncio
    async def test_tool_action_decision(
        self,
        brain,
        sample_state_with_tasks,
        mock_inference_service,
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
        mock_inference_service.generate_structured.return_value = mock_decision

        config = {"configurable": {"thread_id": "test_thread"}}
        result = await brain.think(sample_state_with_tasks, config)

        assert result["decision"]["action_type"] == "tool"
        assert result["decision"]["resource_id"] == "calculator"

    @pytest.mark.asyncio
    async def test_plan_creation_decision(
        self,
        brain,
        sample_state_with_tasks,
        mock_inference_service,
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
        mock_inference_service.generate_structured.return_value = mock_decision

        config = {"configurable": {"thread_id": "test_thread"}}
        result = await brain.think(sample_state_with_tasks, config)

        assert result["decision"]["action_type"] == "plan"
        assert len(result["decision"]["execution_plan"]) == 2
        assert result["decision"]["execution_plan"][0]["phase_id"] == "phase_1"

    @pytest.mark.asyncio
    async def test_finish_decision(
        self,
        brain,
        sample_state_with_tasks,
        mock_inference_service,
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
        mock_inference_service.generate_structured.return_value = mock_decision

        config = {"configurable": {"thread_id": "test_thread"}}
        result = await brain.think(sample_state_with_tasks, config)

        assert result["decision"]["action_type"] == "finish"
        assert result["final_response"] == "The calculation is complete: 2 + 2 = 4"
        assert result["current_task_id"] is None


class TestBrainApprovalHandling:
    """Test human-in-the-loop approval flow."""

    @pytest.mark.asyncio
    async def test_requires_approval_sets_pending(
        self,
        brain,
        sample_state_with_tasks,
        mock_inference_service,
        mock_agent_registry,
        mock_tool_registry,
        mock_content_orchestrator,
    ):
        """Test Brain sets pending_approval when requires_approval=True."""
        mock_decision = BrainDecision(
            action_type="agent",
            resource_id="EmailAgent",
            payload={"instruction": "Send email to finance@example.com"},
            requires_approval=True,
            approval_reason="Will send email with report to finance@example.com",
            reasoning="Sending email requires user approval",
        )
        mock_inference_service.generate_structured.return_value = mock_decision

        config = {"configurable": {"thread_id": "test_thread"}}
        result = await brain.think(sample_state_with_tasks, config)

        assert result["pending_approval"] is True
        assert result["pending_decision"] is not None
        assert result["pending_decision"]["action_type"] == "agent"
        assert result["pending_decision"]["requires_approval"] is True


class TestBrainPhaseManagement:
    """Test phase completion and advancement logic."""

    @pytest.mark.asyncio
    async def test_phase_completion(
        self,
        brain,
        sample_state_with_plan,
        mock_inference_service,
        mock_agent_registry,
        mock_tool_registry,
        mock_content_orchestrator,
    ):
        """Test Brain advances phase when phase_complete=True."""
        mock_decision = BrainDecision(
            action_type="skip",
            phase_complete=True,
            phase_goal_verified="Collected Q3 revenue ($2.1M) and Q4 revenue ($1.8M)",
            reasoning="Phase goal has been met",
        )
        mock_inference_service.generate_structured.return_value = mock_decision

        config = {"configurable": {"thread_id": "test_thread"}}
        result = await brain.think(sample_state_with_plan, config)

        # Current phase should be marked as completed
        updated_plan = result["execution_plan"]
        phase_1 = next(p for p in updated_plan if p["phase_id"] == "phase_1")
        assert phase_1["status"] == "completed"
        assert (
            phase_1["goal_verified"]
            == "Collected Q3 revenue ($2.1M) and Q4 revenue ($1.8M)"
        )

    @pytest.mark.asyncio
    async def test_phase_advancement_to_next(
        self,
        brain,
        sample_state_with_plan,
        mock_inference_service,
        mock_agent_registry,
        mock_tool_registry,
        mock_content_orchestrator,
    ):
        """Test Brain advances to next phase when dependencies are met."""
        # First iteration: complete phase 1
        mock_decision_1 = BrainDecision(
            action_type="skip",
            phase_complete=True,
            phase_goal_verified="Data collected successfully",
            reasoning="Phase 1 complete",
        )
        mock_inference_service.generate_structured.return_value = mock_decision_1

        config = {"configurable": {"thread_id": "test_thread"}}
        result = await brain.think(sample_state_with_plan, config)

        # Should advance to phase 2
        assert result["current_phase_id"] == "phase_2"


class TestBrainInsightsExtraction:
    """Test insights extraction from action results."""

    def test_extract_insights_from_successful_action(self, brain):
        """Test Brain extracts insights from successful action output."""
        state = {
            "execution_result": {
                "success": True,
                "output": {"result": "Q4 revenue: $2.1M, Q3 revenue: $1.8M"},
            },
            "iteration_count": 5,
            "insights": {"existing_key": "existing_value"},
        }

        updated_insights = brain._extract_insights_from_last_action(
            state, state["insights"]
        )

        assert "existing_key" in updated_insights
        assert "step_5" in updated_insights
        assert "Q4 revenue" in updated_insights["step_5"]

    def test_extract_insights_from_string_output(self, brain):
        """Test Brain extracts insights from string output."""
        state = {
            "execution_result": {
                "success": True,
                "output": "The calculation result is 42",
            },
            "iteration_count": 3,
            "insights": {},
        }

        updated_insights = brain._extract_insights_from_last_action(state, {})

        assert "step_3" in updated_insights
        assert "42" in updated_insights["step_3"]

    def test_no_insights_from_failed_action(self, brain):
        """Test Brain doesn't extract insights from failed actions."""
        state = {
            "execution_result": {"success": False, "error": "Agent connection failed"},
            "iteration_count": 2,
            "insights": {},
        }

        updated_insights = brain._extract_insights_from_last_action(state, {})

        assert len(updated_insights) == 0


class TestBrainActionHistoryBuilding:
    """Test action history view construction."""

    def test_build_empty_action_history(self, brain):
        """Test building view with no action history."""
        view = brain._build_action_history_view([])

        assert view == "No actions taken yet."

    def test_build_action_history_below_budget(self, brain):
        """Test building view when history fits in token budget."""
        history = [
            {
                "iteration": 1,
                "action_type": "agent",
                "resource_id": "TestAgent",
                "success": True,
                "result_summary": "Successfully analyzed data",
                "timestamp": time.time(),
                "execution_time_ms": 100,
            },
            {
                "iteration": 2,
                "action_type": "tool",
                "resource_id": "calculator",
                "success": True,
                "result_summary": "Calculation complete: 42",
                "timestamp": time.time(),
                "execution_time_ms": 50,
            },
        ]

        view = brain._build_action_history_view(history, model_context_window=32000)

        assert "[Step 1]" in view
        assert "[Step 2]" in view
        assert "agent:TestAgent" in view
        assert "tool:calculator" in view

    def test_build_action_history_with_truncation(self, brain):
        """Test building view with excessive history (truncation)."""
        # Create 150 actions (more than default budget)
        history = [
            {
                "iteration": i,
                "action_type": "agent",
                "resource_id": f"Agent{i}",
                "success": True,
                "result_summary": f"Result {i}",
                "timestamp": time.time(),
                "execution_time_ms": 100,
            }
            for i in range(1, 151)
        ]

        view = brain._build_action_history_view(history, model_context_window=32000)

        # Should show truncation message with archived count
        assert "archived" in view
        # Should show recent actions
        assert "[Step 150]" in view


class TestBrainPlanValidation:
    """Test execution plan validation."""

    def test_validate_well_formed_plan(self, brain):
        """Test validation of a well-formed execution plan."""
        plan = [
            {
                "phase_id": "phase_1",
                "name": "Phase 1",
                "goal": "Goal 1",
                "depends_on": [],
            },
            {
                "phase_id": "phase_2",
                "name": "Phase 2",
                "goal": "Goal 2",
                "depends_on": ["phase_1"],
            },
        ]

        validated, errors = brain._validate_execution_plan(plan)

        assert len(validated) == 2
        assert len(errors) == 0

    def test_validate_plan_with_missing_phase_id(self, brain):
        """Test validation auto-assigns missing phase IDs."""
        plan = [{"name": "Phase 1", "goal": "Goal 1", "depends_on": []}]

        validated, errors = brain._validate_execution_plan(plan)

        assert len(validated) == 1
        assert validated[0]["phase_id"] == "phase_1"
        assert len(errors) == 1
        assert "Missing phase_id" in errors[0]

    def test_validate_plan_with_invalid_dependency(self, brain):
        """Test validation catches invalid phase dependencies."""
        plan = [
            {
                "phase_id": "phase_1",
                "name": "Phase 1",
                "goal": "Goal 1",
                "depends_on": ["nonexistent_phase"],
            }
        ]

        validated, errors = brain._validate_execution_plan(plan)

        assert len(validated) == 1
        assert any("Invalid dependency" in e for e in errors)


class TestBrainErrorHandling:
    """Test error handling and fallback modes."""

    @pytest.mark.asyncio
    async def test_llm_failure_returns_finish_decision(
        self,
        brain,
        sample_state_with_tasks,
        mock_inference_service,
        mock_agent_registry,
        mock_tool_registry,
        mock_content_orchestrator,
    ):
        """Test Brain handles LLM failure gracefully."""
        # Mock LLM to raise exception
        mock_inference_service.generate_structured.side_effect = Exception(
            "LLM connection failed"
        )

        config = {"configurable": {"thread_id": "test_thread"}}
        result = await brain.think(sample_state_with_tasks, config)

        # When LLM fails, Brain should return a finish decision with final_response
        assert "decision" in result
        assert result["decision"]["action_type"] == "finish"
        assert result["decision"]["is_finished"] is True
        assert "Brain error" in result["decision"]["user_response"]
        # The fallback logic also sets final_response
        assert "final_response" in result or result["decision"]["user_response"]

    @pytest.mark.asyncio
    async def test_max_iterations_exceeded(self, brain, sample_state):
        """Test Brain forces finish when max iterations reached."""
        state = {
            **sample_state,
            "iteration_count": 100,  # Over max_iterations
            "todo_list": [
                {
                    "task_id": "task_1",
                    "description": "Unfinished task",
                    "status": TaskStatus.IN_PROGRESS,
                    "priority": 10,
                    "payload": {},
                }
            ],
        }

        result = await brain.think(state)

        assert result["final_response"] is not None
        assert "Maximum iterations reached" in result["final_response"]

    @pytest.mark.asyncio
    async def test_max_failures_triggers_fallback(self, brain, sample_state):
        """Test Brain enters fallback mode after consecutive failures."""
        state = {
            **sample_state,
            "failure_count": 5,  # Over max_failures
            "memory": {"key": "value"},
            "insights": {"insight": "Important learning"},
        }

        result = await brain.think(state)

        # In fallback mode, Brain returns a finish decision with user_response
        assert "decision" in result
        assert result["decision"]["action_type"] == "finish"
        # The final_response is set for both the decision and as a top-level field
        assert result["decision"].get("user_response") or result.get("final_response")
        if result.get("final_response"):
            assert "multiple issues" in result["final_response"]
        else:
            assert "multiple issues" in result["decision"].get("user_response", "")


class TestBrainTodoPreview:
    """Test todo list preview construction."""

    def test_build_empty_todo_preview(self, brain):
        """Test building preview with empty todo list."""
        preview = brain._build_todo_preview([])

        assert preview == "Empty"

    def test_build_todo_preview_with_tasks(self, brain):
        """Test building preview with tasks."""
        todo_list = [
            {
                "task_id": "task_1",
                "description": "Analyze Q3 data",
                "status": TaskStatus.PENDING,
                "priority": 10,
                "payload": {},
            },
            {
                "task_id": "task_2",
                "description": "Create report",
                "status": TaskStatus.IN_PROGRESS,
                "priority": 5,
                "payload": {},
            },
            {
                "task_id": "task_3",
                "description": "Send email",
                "status": TaskStatus.COMPLETED,
                "priority": 1,
                "payload": {},
            },
        ]

        preview = brain._build_todo_preview(todo_list)

        assert "[PENDING]" in preview
        assert "[IN_PROGRESS]" in preview
        assert "[COMPLETED]" in preview
        assert "task_1" in preview
        assert "Analyze Q3 data" in preview
