import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import sys
import os

from schemas import AgentResponseStatus, OrchestratorMessage
from agents.mail_agent.schemas import EmailAction
from agents.mail_agent.agent import execute_action

@pytest.mark.asyncio
async def test_archive_action_execution():
    # Mock the LLM client to return an "archive" step
    with patch('agents.mail_agent.agent.llm_client') as mock_llm_client:
        mock_llm_client.decompose_complex_request = AsyncMock(return_value={
            "steps": [
                {
                    "action": "archive emails",
                    "params": {"message_ids": ["123"], "user_id": "me"}
                }
            ]
        })

        # Mock manage_emails to just return success and capture arguments
        with patch('agents.mail_agent.agent.manage_emails', new_callable=AsyncMock) as mock_manage:
            mock_manage.return_value = MagicMock(success=True, result="Archived")
            
            # Execute the action
            msg = OrchestratorMessage(
                action=None, 
                payload={"prompt": "Archive emails from John", "task_id": "test_task"},
                source="user", # Required fields? check schema
                target="mail_agent"
            )
            # OrchestratorMessage might have defaults or optional fields.
            
            response = await execute_action(msg)

            # Assertions
            if response.status == AgentResponseStatus.ERROR:
                 print(f"FAILED: {response.error}")

            assert response.status == AgentResponseStatus.COMPLETE
            mock_manage.assert_called_once()
            call_args = mock_manage.call_args[0][0] # First arg is the request
            assert call_args.action == EmailAction.ARCHIVE
            assert call_args.message_ids == ["123"]

@pytest.mark.asyncio
async def test_bad_label_archive_execution():
    # Mock LLM to return "add label Archive"
    with patch('agents.mail_agent.agent.llm_client') as mock_llm_client:
        mock_llm_client.decompose_complex_request = AsyncMock(return_value={
            "steps": [
                {
                    "action": "add label",
                    "params": {"message_ids": ["456"], "labels": ["Archive"], "user_id": "me"}
                }
            ]
        })

        with patch('agents.mail_agent.agent.manage_emails', new_callable=AsyncMock) as mock_manage:
            mock_manage.return_value = MagicMock(success=True, result="Archived (Corrected)")

            # Execute
            msg = OrchestratorMessage(
                action=None,
                payload={"prompt": "Label emails as Archive", "task_id": "test_task_2"},
                source="user",
                target="mail_agent"
            )
            response = await execute_action(msg)

            if response.status == AgentResponseStatus.ERROR:
                 print(f"FAILED: {response.error}")

            assert response.status == AgentResponseStatus.COMPLETE
            mock_manage.assert_called_once()
            call_args = mock_manage.call_args[0][0]
            assert call_args.action == EmailAction.ARCHIVE
            assert call_args.message_ids == ["456"]
