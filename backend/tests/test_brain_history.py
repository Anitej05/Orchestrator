
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from langchain_core.messages import HumanMessage, AIMessage

backend_root = Path(__file__).parent.parent
sys.path.insert(0, str(backend_root))

# Mock inference service
mock_inference_service_impl = Mock()
mock_inference_service_impl.generate_structured = AsyncMock()
mock_inference_service_impl.InferencePriority = Mock()
mock_inference_service_impl.InferencePriority.SPEED = "speed"

# Patch dependencies
with (
    patch("services.inference_service.inference_service", mock_inference_service_impl),
    patch("orchestrator.brain.inference_service", mock_inference_service_impl),
):
    from backend.orchestrator.brain import Brain, BrainDecision

@pytest.fixture
def brain():
    return Brain()

@pytest.mark.asyncio
async def test_brain_includes_conversation_history(brain):
    """Test that brain.think includes conversation history in the prompt."""
    
    # Mock state with conversation history
    state = {
        "original_prompt": "Test Context",
        "messages": [
            HumanMessage(content="Hello Brain"),
            AIMessage(content="Hello User"),
            HumanMessage(content="What is my name?"),
        ],
        "todo_list": [{"task_id": "1", "description": "test", "status": "pending"}],
        "memory": {},
        "insights": {},
        "action_history": [],
    }
    
    # Mock return value for generate_structured
    mock_inference_service_impl.generate_structured.return_value = BrainDecision(
        action_type="finish",
        user_response="Test response"
    )
    
    # Run think
    await brain.think(state)
    
    # Verify prompt content
    call_args = mock_inference_service_impl.generate_structured.call_args
    assert call_args is not None
    
    # Extract prompt from call arguments (kwargs['messages'][0].content)
    messages = call_args.kwargs.get("messages")
    prompt = messages[0].content
    
    # Assertions
    assert "## CONVERSATION HISTORY" in prompt
    assert "User: Hello Brain" in prompt
    assert "Assistant: Hello User" in prompt

def test_apply_decision_handles_none_payload(brain):
    """Test that _apply_decision_to_state converts None payload to empty dict."""
    state = {"todo_list": [], "iteration_count": 0}
    
    # Create decision with explicitly None payload
    # Note: Pydantic might default to {} if we use default_factory, 
    # so we must force it to None for this test if generic instantiation allows it,
    # or rely on the fact that we changed schema to Optional.
    
    decision = BrainDecision(
        action_type="tool",
        resource_id="test_tool",
        payload=None  # This is now allowed by schema schema
    )
    
    # Apply to state
    updates = brain._apply_decision_to_state(state, decision)
    
    # Verify payload is {} in updates
    decision_dump = updates["decision"]
    assert decision_dump["payload"] is not None
    assert decision_dump["payload"] == {}
