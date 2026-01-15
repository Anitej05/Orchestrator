import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import sys
import os

from schemas import AgentResponseStatus, OrchestratorMessage
from agents.mail_agent.schemas import EmailAction
from agents.mail_agent.agent import execute_action

@pytest.mark.asyncio
async def test_error_feedback_loop():
    """
    Test that the agent retries with error context when a step fails.
    Scenario:
    1. LLM proposes "add label Archive" (invalid).
    2. Execution fails (mocked).
    3. Loop catches error, calls LLM again with error context.
    4. LLM proposes "archive emails" (valid).
    5. Execution succeeds.
    """
    
    # Mock LLM client
    with patch('agents.mail_agent.agent.llm_client') as mock_llm_client:
        # Side effect for decompose: return bad plan first, then good plan
        mock_llm_client.decompose_complex_request = AsyncMock(side_effect=[
            # Attempt 1: Bad Plan
            {
                "steps": [
                    {
                        "action": "add_labels", 
                        "params": {"message_ids": ["123"], "labels": ["Archive"], "user_id": "me"}
                    }
                ]
            },
            # Attempt 2: Good Plan (after receiving error)
            {
                "steps": [
                    {
                        "action": "archive_emails",
                        "params": {"message_ids": ["123"], "user_id": "me"}
                    }
                ]
            }
        ])

        # Mock manage_emails
        with patch('agents.mail_agent.agent.manage_emails', new_callable=AsyncMock) as mock_manage:
            # Side effect: first call fails, second call succeeds
            # note: manage_emails returns a GmailResponse object
            failure_response = MagicMock(success=False, error="Label 'Archive' not found")
            success_response = MagicMock(success=True, result="Archived successfully")
            
            mock_manage.side_effect = [failure_response, success_response] # First call fails, second succeeds

            # Execute
            msg = OrchestratorMessage(
                action=None,
                payload={"prompt": "Archive emails from John", "task_id": "test_retry_loop"},
                source="user",
                target="mail_agent"
            )
            
            response = await execute_action(msg)

            # Assertions
            assert response.status == AgentResponseStatus.COMPLETE
            
            # Verify decompose was called twice
            assert mock_llm_client.decompose_complex_request.call_count == 2
            
            # Verify second call had error context
            args, kwargs = mock_llm_client.decompose_complex_request.call_args_list[1]
            assert "Label 'Archive' not found" in kwargs['error_context']
            
            # Verify manage_emails called twice
            assert mock_manage.call_count == 2
            
            # Verify last call was correct
            final_call_args = mock_manage.call_args[0][0]
            assert final_call_args.action == EmailAction.ARCHIVE
