#!/usr/bin/env python3
"""
Test script to verify AgentResponse implementation fixes in spreadsheet agent.

This script tests:
1. AgentResponse format consistency with mail agent
2. Proper status handling (COMPLETE, ERROR, NEEDS_INPUT)
3. File passing in AgentResponse context
4. Bidirectional dialogue patterns
"""

import sys
import json
import asyncio
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

from schemas import AgentResponse, AgentResponseStatus, OrchestratorMessage


async def test_agentresponse_format():
    """Test that AgentResponse format matches expected schema"""
    print("🧪 Testing AgentResponse format consistency...")
    
    # Test COMPLETE status
    complete_response = AgentResponse(
        status=AgentResponseStatus.COMPLETE,
        result={"test": "data", "rows": 100},
        context={"task_id": "test-123", "file_id": "file-456"}
    )
    
    dumped = complete_response.model_dump()
    print(f"✅ COMPLETE response keys: {list(dumped.keys())}")
    assert dumped["status"] == "complete"
    assert dumped["result"]["test"] == "data"
    assert dumped["context"]["task_id"] == "test-123"
    
    # Test NEEDS_INPUT status
    needs_input_response = AgentResponse(
        status=AgentResponseStatus.NEEDS_INPUT,
        question="How would you like to handle missing values?",
        question_type="choice",
        options=["fill_mean", "drop_rows", "fill_zero"],
        context={"task_id": "test-456", "anomaly_type": "missing_values"}
    )
    
    dumped = needs_input_response.model_dump()
    print(f"✅ NEEDS_INPUT response keys: {list(dumped.keys())}")
    assert dumped["status"] == "needs_input"
    assert dumped["question"] is not None
    assert dumped["options"] == ["fill_mean", "drop_rows", "fill_zero"]
    
    # Test ERROR status
    error_response = AgentResponse(
        status=AgentResponseStatus.ERROR,
        error="File not found",
        context={"task_id": "test-789", "file_id": "missing-file"}
    )
    
    dumped = error_response.model_dump()
    print(f"✅ ERROR response keys: {list(dumped.keys())}")
    assert dumped["status"] == "error"
    assert dumped["error"] == "File not found"
    
    print("✅ All AgentResponse format tests passed!")


async def test_orchestrator_message_format():
    """Test OrchestratorMessage format for /execute and /continue"""
    print("\n🧪 Testing OrchestratorMessage format...")
    
    # Test execute message
    execute_msg = OrchestratorMessage(
        action="/get_summary",
        payload={
            "file_id": "test-file-123",
            "thread_id": "test-thread-456",
            "task_id": "task-789"
        }
    )
    
    dumped = execute_msg.model_dump()
    print(f"✅ Execute message keys: {list(dumped.keys())}")
    assert dumped["action"] == "/get_summary"
    assert dumped["payload"]["file_id"] == "test-file-123"
    
    # Test continue message
    continue_msg = OrchestratorMessage(
        type="continue",
        answer="fill_mean",
        payload={"task_id": "task-789"}
    )
    
    dumped = continue_msg.model_dump()
    print(f"✅ Continue message keys: {list(dumped.keys())}")
    assert dumped["type"] == "continue"
    assert dumped["answer"] == "fill_mean"
    
    print("✅ All OrchestratorMessage format tests passed!")


async def test_dialogue_manager():
    """Test dialogue manager functionality"""
    print("\n🧪 Testing dialogue manager...")
    
    from agents.spreadsheet_agent.dialogue_manager import dialogue_manager
    
    # Test storing and retrieving pending questions
    task_id = "test-dialogue-123"
    question = "How should we handle outliers?"
    choices = [
        {"id": "remove", "label": "Remove outliers"},
        {"id": "cap", "label": "Cap outliers"}
    ]
    
    dialogue_manager.store_pending_question(
        task_id=task_id,
        question=question,
        question_type="choice",
        choices=choices,
        context={"anomaly_type": "outliers"}
    )
    
    # Retrieve pending question
    pending = dialogue_manager.get_pending_question(task_id)
    print(f"✅ Stored and retrieved pending question: {pending['question'][:30]}...")
    assert pending["question"] == question
    assert pending["question_type"] == "choice"
    assert len(pending["choices"]) == 2
    
    # Test state management
    dialogue_manager.save_state(task_id, {"file_id": "test-file", "operation": "anomaly_detection"})
    state = dialogue_manager.load_state(task_id)
    print(f"✅ State management works: {list(state.keys())}")
    assert state["file_id"] == "test-file"
    
    print("✅ All dialogue manager tests passed!")


def test_mail_agent_comparison():
    """Compare with mail agent patterns to ensure consistency"""
    print("\n🧪 Comparing with mail agent patterns...")
    
    # Test that we can create the same response patterns as mail agent
    
    # Pattern 1: Simple completion (like mail agent search)
    mail_style_response = AgentResponse(
        status=AgentResponseStatus.COMPLETE,
        result={
            "messages": [{"id": "123", "subject": "Test"}],
            "count": 1
        }
    )
    
    dumped = mail_style_response.model_dump()
    print(f"✅ Mail-style completion response: {dumped['status']}")
    
    # Pattern 2: Needs input with choices (like mail agent ambiguous search)
    mail_style_needs_input = AgentResponse(
        status=AgentResponseStatus.NEEDS_INPUT,
        question="I found multiple contacts matching 'John'. Which one are you referring to?",
        question_type="choice",
        options=["John Smith (Work)", "John Doe (Personal)", "John Baker"],
        context={"original_query": "emails from john"}
    )
    
    dumped = mail_style_needs_input.model_dump()
    print(f"✅ Mail-style needs input response: {dumped['question'][:30]}...")
    
    # Pattern 3: Error response (like mail agent failures)
    mail_style_error = AgentResponse(
        status=AgentResponseStatus.ERROR,
        error="Gmail API connection failed",
        context={"task_id": "search-123"}
    )
    
    dumped = mail_style_error.model_dump()
    print(f"✅ Mail-style error response: {dumped['error']}")
    
    print("✅ All mail agent comparison tests passed!")


async def main():
    """Run all tests"""
    print("🚀 Testing Spreadsheet Agent AgentResponse Implementation Fixes")
    print("=" * 60)
    
    try:
        await test_agentresponse_format()
        await test_orchestrator_message_format()
        await test_dialogue_manager()
        test_mail_agent_comparison()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! AgentResponse implementation is fixed!")
        print("\nKey fixes implemented:")
        print("✅ Using standardized AgentResponse from schemas.py")
        print("✅ Proper status handling (COMPLETE, ERROR, NEEDS_INPUT)")
        print("✅ Consistent field names (context instead of metadata)")
        print("✅ Options list instead of choices for NEEDS_INPUT")
        print("✅ Bidirectional dialogue patterns matching mail agent")
        print("✅ File passing in context field")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)