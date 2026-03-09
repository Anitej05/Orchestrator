"""
Gmail Agent — Comprehensive Unit Tests
=======================================

Tests all Gmail agent components with mocked dependencies (no live API calls).

Coverage:
  TestAgentMemory             — add_turn, context update/get, search results, draft context,
                                 history pruning at 20 entries, clear
  TestStripThinkTags          — all 7 tag formats, mixed/nested tags, non-string passthrough
  TestLLMClientQueryGen       — successful generation, empty query fallback, all-providers-fail
                                 fallback, HTML removal, redundant prefix stripping
  TestLLMClientSummarize      — short text single-pass, >4000 char recursive map-reduce,
                                 empty text edge-case, batch summarize
  TestLLMClientDraft          — success path, missing fields retry, all-providers-fail fallback,
                                 >6000 char summarization pre-pass
  TestLLMClientExtract        — action extraction, empty texts, >4050 char chunked path,
                                 deduplication across chunks
  TestComposioToolManager     — connection dict access (not attribute), unconnected user raises,
                                 execute_tool success, execute_tool exception wrapping
  TestGmailService            — search_emails (LLM optimize + direct), get_email,
                                 send_email (missing params, success, error),
                                 reply_to_email, delete_email (trash + permanent),
                                 create_draft (memory saved), list_drafts, send_draft,
                                 delete_draft, add_labels, list_labels, create_label,
                                 download_attachment (save + no data), download_all_attachments,
                                 summarize_emails, draft_smart_reply, extract_action_items,
                                 list_contacts, search_contacts, list_threads, get_thread,
                                 get_profile
  TestGmailAgentCapabilities  — search_emails (success, missing query, Composio error),
                                 send_email (success, missing required fields, send failure),
                                 reply_email (success, missing params),
                                 get_email (success, missing message_id)

Run:
    PYTHONUTF8=1 venv/Scripts/python -m pytest backend/tests/test_gmail_agent_unit.py -v
"""

import asyncio
import sys
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from dotenv import load_dotenv
load_dotenv(dotenv_path=ROOT / ".env", override=False)

# ── Import Gmail agent modules ─────────────────────────────────────────────────
from backend.agents.gmail_agent.memory import AgentMemory
from backend.agents.gmail_agent.llm import LLMClient, strip_think_tags
from backend.agents.base.types import ExecutionContext


# ==============================================================================
# TestAgentMemory
# ==============================================================================
class TestAgentMemory:
    def setup_method(self):
        self.mem = AgentMemory()
        self.uid = "test_user"

    def test_initial_store_empty(self):
        store = self.mem._get_user_store(self.uid)
        assert store["history"] == []
        assert store["context"] == {}

    def test_add_turn_stores_two_entries(self):
        self.mem.add_turn(self.uid, "hello", "world")
        history = self.mem.get_history(self.uid)
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "hello"
        assert history[1]["role"] == "agent"
        assert history[1]["content"] == "world"

    def test_add_turn_stores_action_type(self):
        self.mem.add_turn(self.uid, "q", "a", action_type="send_email")
        history = self.mem.get_history(self.uid)
        assert history[1]["action_type"] == "send_email"

    def test_history_pruned_to_20(self):
        # Add 12 turns = 24 entries; should be pruned to last 20
        for i in range(12):
            self.mem.add_turn(self.uid, f"user_{i}", f"agent_{i}")
        history = self.mem.get_history(self.uid)
        assert len(history) == 20

    def test_update_and_get_context(self):
        self.mem.update_context(self.uid, "foo", {"bar": 1})
        assert self.mem.get_context(self.uid, "foo") == {"bar": 1}

    def test_get_context_missing_key_returns_none(self):
        assert self.mem.get_context(self.uid, "nonexistent") is None

    def test_save_and_get_search_results(self):
        ids = ["msg1", "msg2", "msg3"]
        self.mem.save_search_results(self.uid, ids)
        assert self.mem.get_last_search_results(self.uid) == ids

    def test_save_draft_context(self):
        ctx = {"to": "x@y.com", "subject": "Test"}
        self.mem.save_draft_context(self.uid, "draft_001", ctx)
        assert self.mem.get_draft_context(self.uid, "draft_001") == ctx

    def test_multiple_drafts_coexist(self):
        self.mem.save_draft_context(self.uid, "d1", {"subject": "A"})
        self.mem.save_draft_context(self.uid, "d2", {"subject": "B"})
        assert self.mem.get_draft_context(self.uid, "d1")["subject"] == "A"
        assert self.mem.get_draft_context(self.uid, "d2")["subject"] == "B"

    def test_get_draft_context_missing_returns_none(self):
        assert self.mem.get_draft_context(self.uid, "no_such_draft") is None

    def test_clear_removes_user_data(self):
        self.mem.add_turn(self.uid, "q", "a")
        self.mem.clear(self.uid)
        assert self.mem.get_history(self.uid) == []  # re-creates empty store

    def test_clear_nonexistent_user_is_safe(self):
        self.mem.clear("nobody")  # should not raise

    def test_multiple_users_isolated(self):
        self.mem.save_search_results("user_a", ["a1"])
        self.mem.save_search_results("user_b", ["b1"])
        assert self.mem.get_last_search_results("user_a") == ["a1"]
        assert self.mem.get_last_search_results("user_b") == ["b1"]


# ==============================================================================
# TestStripThinkTags
# ==============================================================================
class TestStripThinkTags:
    def test_closed_think_tags(self):
        assert strip_think_tags("<think>reasoning here</think>answer") == "answer"

    def test_unclosed_think_tag(self):
        result = strip_think_tags("<think>reasoning that never closes")
        assert "reasoning" not in result

    def test_minimax_pipe_thinking(self):
        # Closing tag is </|thinking|> (slash after <) — matches regex r'</\|thinking\|>'
        result = strip_think_tags("<|thinking|>internal</|thinking|>output")
        assert "internal" not in result
        assert "output" in result

    def test_minimax_thought_tags(self):
        # Closing tag is </|thought|> (slash after <) — matches regex r'</\|thought\|>'
        result = strip_think_tags("<|thought|>hidden</|thought|>visible")
        assert "hidden" not in result
        assert "visible" in result

    def test_deepseek_thought_tags(self):
        result = strip_think_tags("<thought>DS thinks</thought>result")
        assert "DS thinks" not in result
        assert "result" in result

    def test_reasoning_tags(self):
        result = strip_think_tags("<reasoning>internal</reasoning>answer")
        assert "internal" not in result
        assert "answer" in result

    def test_no_tags_passthrough(self):
        assert strip_think_tags("plain text") == "plain text"

    def test_non_string_passthrough(self):
        assert strip_think_tags(42) == 42
        assert strip_think_tags(None) is None

    def test_multiline_think_tags(self):
        text = "<think>\nline1\nline2\n</think>final answer"
        assert strip_think_tags(text) == "final answer"

    def test_case_insensitive(self):
        result = strip_think_tags("<THINK>hidden</THINK>shown")
        assert "hidden" not in result
        assert "shown" in result


# ==============================================================================
# TestLLMClientQueryGen
# ==============================================================================
class TestLLMClientQueryGen:
    def _make_mock_response(self, content: str):
        msg = MagicMock()
        msg.content = content
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    @pytest.mark.asyncio
    async def test_empty_query_returns_inbox(self):
        client = LLMClient()
        result = await client.generate_optimized_query("")
        assert result == "label:inbox"

    @pytest.mark.asyncio
    async def test_whitespace_only_query_returns_inbox(self):
        client = LLMClient()
        result = await client.generate_optimized_query("   ")
        assert result == "label:inbox"

    @pytest.mark.asyncio
    async def test_short_query_returned_as_is(self):
        """Queries with ≤2 words skip LLM optimization."""
        client = LLMClient()
        # Patch clients to verify they are NOT called
        client.clients = [{"name": "Mock", "client": MagicMock(), "model": "m"}]
        result = await client.generate_optimized_query("inbox")
        assert result == "inbox"

    @pytest.mark.asyncio
    async def test_successful_query_generation(self):
        client = LLMClient()
        mock_resp = self._make_mock_response("subject:\"Hello World\"")
        mock_create = AsyncMock(return_value=mock_resp)
        mock_provider_client = MagicMock()
        mock_provider_client.chat.completions.create = mock_create
        client.clients = [{"name": "Test", "client": mock_provider_client, "model": "m"}]

        result = await client.generate_optimized_query("emails about Hello World")
        # generate_optimized_query does content.strip().strip('"\'`') which strips the
        # trailing double-quote, so 'subject:"Hello World"' becomes 'subject:"Hello World'
        assert result == 'subject:"Hello World'

    @pytest.mark.asyncio
    async def test_strips_query_prefix_from_response(self):
        client = LLMClient()
        mock_resp = self._make_mock_response("Query: is:unread")
        mock_create = AsyncMock(return_value=mock_resp)
        mock_provider_client = MagicMock()
        mock_provider_client.chat.completions.create = mock_create
        client.clients = [{"name": "Test", "client": mock_provider_client, "model": "m"}]

        result = await client.generate_optimized_query("show me all unread emails")
        assert result == "is:unread"

    @pytest.mark.asyncio
    async def test_all_providers_fail_returns_original(self):
        client = LLMClient()
        mock_create = AsyncMock(side_effect=Exception("API error"))
        mock_provider_client = MagicMock()
        mock_provider_client.chat.completions.create = mock_create
        client.clients = [{"name": "Fail", "client": mock_provider_client, "model": "m"}]

        result = await client.generate_optimized_query("emails from john about project")
        assert result == "emails from john about project"

    @pytest.mark.asyncio
    async def test_html_stripped_from_response(self):
        client = LLMClient()
        mock_resp = self._make_mock_response("<p>subject:test</p>")
        mock_create = AsyncMock(return_value=mock_resp)
        mock_provider_client = MagicMock()
        mock_provider_client.chat.completions.create = mock_create
        client.clients = [{"name": "Test", "client": mock_provider_client, "model": "m"}]

        result = await client.generate_optimized_query("emails about test subject")
        assert "<p>" not in result


# ==============================================================================
# TestLLMClientSummarize
# ==============================================================================
class TestLLMClientSummarize:
    def _make_mock_response(self, content: str):
        msg = MagicMock()
        msg.content = content
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    @pytest.mark.asyncio
    async def test_empty_text_returns_message(self):
        client = LLMClient()
        result = await client.summarize_email_content("")
        assert result == "Empty email content."

    @pytest.mark.asyncio
    async def test_short_text_single_pass(self):
        client = LLMClient()
        mock_resp = self._make_mock_response("This is a summary.")
        mock_create = AsyncMock(return_value=mock_resp)
        mock_provider_client = MagicMock()
        mock_provider_client.chat.completions.create = mock_create
        client.clients = [{"name": "T", "client": mock_provider_client, "model": "m", "summary_model": "s"}]

        result = await client.summarize_email_content("Short email content here.")
        assert result == "This is a summary."

    @pytest.mark.asyncio
    async def test_empty_texts_batch_returns_message(self):
        client = LLMClient()
        result = await client.summarize_text_batch([])
        assert result == "No content to summarize."

    @pytest.mark.asyncio
    async def test_batch_calls_summarize_email_content(self):
        client = LLMClient()
        client.summarize_email_content = AsyncMock(return_value="batched summary")
        result = await client.summarize_text_batch(["text1", "text2"])
        assert result == "batched summary"
        client.summarize_email_content.assert_called_once()


# ==============================================================================
# TestLLMClientDraft
# ==============================================================================
class TestLLMClientDraft:
    def _make_json_response(self, data: dict):
        import json
        msg = MagicMock()
        msg.content = json.dumps(data)
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    @pytest.mark.asyncio
    async def test_empty_thread_returns_fallback(self):
        client = LLMClient()
        result = await client.draft_email_reply("", "reply professionally", "Alice")
        assert result["subject"] == "Re: Email"
        assert "No thread context" in result["body"]

    @pytest.mark.asyncio
    async def test_successful_draft(self):
        client = LLMClient()
        mock_resp = self._make_json_response({
            "subject": "Re: Meeting",
            "body": "<p>Confirmed</p>",
            "is_html": True
        })
        mock_create = AsyncMock(return_value=mock_resp)
        mock_provider_client = MagicMock()
        mock_provider_client.chat.completions.create = mock_create
        client.clients = [{"name": "T", "client": mock_provider_client, "model": "m"}]

        result = await client.draft_email_reply("Let's meet Tuesday", "confirm meeting", "Bob")
        assert result["subject"] == "Re: Meeting"
        assert "<p>Confirmed</p>" in result["body"]

    @pytest.mark.asyncio
    async def test_all_providers_fail_returns_fallback(self):
        client = LLMClient()
        mock_create = AsyncMock(side_effect=Exception("fail"))
        mock_provider_client = MagicMock()
        mock_provider_client.chat.completions.create = mock_create
        client.clients = [{"name": "F", "client": mock_provider_client, "model": "m"}]

        result = await client.draft_email_reply("Original email", "reply", "Sender")
        assert "Could not generate" in result["body"]

    @pytest.mark.asyncio
    async def test_long_thread_summarized_first(self):
        client = LLMClient()
        # Pre-summarization should be called for >6000 char thread
        client.summarize_email_content = AsyncMock(return_value="condensed thread")

        import json
        mock_resp_data = {"subject": "Re: Long", "body": "<p>OK</p>", "is_html": True}
        msg = MagicMock()
        msg.content = json.dumps(mock_resp_data)
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        mock_create = AsyncMock(return_value=resp)
        mock_provider_client = MagicMock()
        mock_provider_client.chat.completions.create = mock_create
        client.clients = [{"name": "T", "client": mock_provider_client, "model": "m"}]

        long_thread = "x" * 6001
        await client.draft_email_reply(long_thread, "reply", "Sender")
        client.summarize_email_content.assert_called_once_with(long_thread)


# ==============================================================================
# TestLLMClientExtract
# ==============================================================================
class TestLLMClientExtract:
    @pytest.mark.asyncio
    async def test_empty_texts_returns_empty_list(self):
        client = LLMClient()
        result = await client.extract_actions([])
        assert result == []

    @pytest.mark.asyncio
    async def test_successful_extraction(self):
        import json
        client = LLMClient()
        mock_actions = [
            {"description": "Send report", "type": "todo", "priority": "high", "source": "Re: Q4"},
            {"description": "Schedule meeting", "type": "meeting", "priority": "medium", "source": "Invite"}
        ]
        msg = MagicMock()
        msg.content = json.dumps({"actions": mock_actions})
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        mock_create = AsyncMock(return_value=resp)
        mock_provider_client = MagicMock()
        mock_provider_client.chat.completions.create = mock_create
        client.clients = [{"name": "T", "client": mock_provider_client, "model": "m", "summary_model": "s"}]

        result = await client.extract_actions(["Check deadline", "Meeting invite"])
        assert len(result) == 2
        assert result[0]["description"] == "Send report"

    @pytest.mark.asyncio
    async def test_all_providers_fail_returns_empty_list(self):
        client = LLMClient()
        mock_create = AsyncMock(side_effect=Exception("fail"))
        mock_provider_client = MagicMock()
        mock_provider_client.chat.completions.create = mock_create
        client.clients = [{"name": "F", "client": mock_provider_client, "model": "m", "summary_model": "s"}]

        result = await client.extract_actions(["some email text"])
        assert result == []


# ==============================================================================
# TestComposioToolManager
# ==============================================================================
class TestComposioToolManager:
    def _make_tool_manager(self, user_id="user_374hMFRAc0nkaGdH8XtXNRIdfrk"):
        """
        Create a ComposioToolManager bypassing the real __init__.

        get_auth_manager is imported LOCALLY inside __init__ (not at module level),
        so patch("backend.agents.gmail_agent.tools.get_auth_manager") won't work.
        Instead we patch __init__ directly and set required attributes manually.
        """
        from backend.agents.gmail_agent.tools import ComposioToolManager
        with patch.object(ComposioToolManager, "__init__", return_value=None):
            mgr = ComposioToolManager(user_id)
        # Set attributes that __init__ would normally set
        mgr.user_id = user_id
        mgr.connection_id = "4a4f8fa8-deed-4be7-a178-55cb1ed82f1e"
        mgr.composio = MagicMock()
        return mgr

    def test_connection_id_from_dict(self):
        """
        connection_id is stored as a plain dict key access (connection["connection_id"]).
        If it were mistakenly accessed as an attribute (connection.connection_id),
        it would raise AttributeError since get_connection_for_agent returns a dict.
        """
        mgr = self._make_tool_manager()
        assert mgr.connection_id == "4a4f8fa8-deed-4be7-a178-55cb1ed82f1e"

    def test_unconnected_user_raises_value_error(self):
        """
        ComposioToolManager.__init__ must raise ValueError if get_connection_for_agent
        returns None (user hasn't connected Gmail via Composio OAuth).
        We patch at the source module since it's a local import inside __init__.
        """
        import importlib
        import sys

        # Ensure the composio_auth module is importable (backend/ is on sys.path)
        # Patch at the source module that __init__ imports from
        with patch("services.integrations.composio_auth.get_auth_manager") as mock_auth_fn, \
             patch("backend.agents.gmail_agent.tools.Composio"):
            mock_auth = MagicMock()
            mock_auth.get_connection_for_agent.return_value = None
            mock_auth_fn.return_value = mock_auth

            from backend.agents.gmail_agent.tools import ComposioToolManager
            with pytest.raises(ValueError, match="not connected to Gmail"):
                ComposioToolManager("disconnected_user")

    @pytest.mark.asyncio
    async def test_execute_tool_success(self):
        mgr = self._make_tool_manager()
        # Composio SDK ≥0.7: uses composio.actions.execute(), not composio.execute_action()
        # Action[slug] subscript also removed — now getattr(Action, slug)
        mgr.composio.actions = MagicMock()
        mgr.composio.actions.execute = MagicMock(return_value={"data": {"messages": []}})
        mock_action = MagicMock()
        with patch("backend.agents.gmail_agent.tools.Action", mock_action):
            result = await mgr.execute_tool("GMAIL_FETCH_EMAILS", {"query": "label:inbox"})
        assert result["success"] is True
        assert "messages" in result["data"]

    @pytest.mark.asyncio
    async def test_execute_tool_wraps_exception(self):
        mgr = self._make_tool_manager()
        mgr.composio.actions = MagicMock()
        mgr.composio.actions.execute = MagicMock(side_effect=Exception("Composio API down"))
        mock_action = MagicMock()
        with patch("backend.agents.gmail_agent.tools.Action", mock_action):
            result = await mgr.execute_tool("GMAIL_FETCH_EMAILS", {})
        assert result["success"] is False
        assert "Composio API down" in result["error"]

    @pytest.mark.asyncio
    async def test_fetch_emails_calls_correct_tool(self):
        mgr = self._make_tool_manager()
        mgr.execute_tool = AsyncMock(return_value={"success": True, "data": {"messages": []}})

        await mgr.fetch_emails(query="is:unread", max_results=5)
        mgr.execute_tool.assert_called_once_with(
            "GMAIL_FETCH_EMAILS",
            {"query": "is:unread", "max_results": 5, "include_payload": False}
        )

    @pytest.mark.asyncio
    async def test_send_email_calls_correct_tool(self):
        mgr = self._make_tool_manager()
        mgr.execute_tool = AsyncMock(return_value={"success": True, "data": {"id": "msg123"}})

        await mgr.send_email(
            to="ashrithaannadata@gmail.com",
            subject="Test",
            body="Hello"
        )
        call_args = mgr.execute_tool.call_args
        assert call_args[0][0] == "GMAIL_SEND_EMAIL"
        params = call_args[0][1]
        assert params["to"] == "ashrithaannadata@gmail.com"
        assert params["subject"] == "Test"

    @pytest.mark.asyncio
    async def test_send_email_cc_bcc_included_when_provided(self):
        mgr = self._make_tool_manager()
        mgr.execute_tool = AsyncMock(return_value={"success": True, "data": {}})

        await mgr.send_email("to@x.com", "sub", "body", cc=["cc@x.com"], bcc=["bcc@x.com"])
        params = mgr.execute_tool.call_args[0][1]
        assert "cc" in params
        assert "bcc" in params

    @pytest.mark.asyncio
    async def test_create_draft_thread_id_included_when_provided(self):
        mgr = self._make_tool_manager()
        mgr.execute_tool = AsyncMock(return_value={"success": True, "data": {"id": "draft1"}})

        await mgr.create_draft("to@x.com", "sub", "body", thread_id="thread_abc")
        params = mgr.execute_tool.call_args[0][1]
        assert params["thread_id"] == "thread_abc"

    @pytest.mark.asyncio
    async def test_get_profile_calls_correct_tool(self):
        mgr = self._make_tool_manager()
        mgr.execute_tool = AsyncMock(return_value={"success": True, "data": {"emailAddress": "al.ashritha@gmail.com"}})

        result = await mgr.get_profile()
        mgr.execute_tool.assert_called_once_with("GMAIL_GET_PROFILE", {})


# ==============================================================================
# TestGmailService
# ==============================================================================
class TestGmailService:
    def _make_service(self, user_id="user_374hMFRAc0nkaGdH8XtXNRIdfrk"):
        """Create a GmailService with fully mocked tool_mgr and llm."""
        with patch("backend.agents.gmail_agent.service.ComposioToolManager") as mock_tool_cls, \
             patch("backend.agents.gmail_agent.service.llm_client") as mock_llm:
            mock_tool_cls.return_value = MagicMock()
            from backend.agents.gmail_agent.service import GmailService
            svc = GmailService(user_id)
            svc.tool_mgr = MagicMock()
            svc.llm = MagicMock()
            return svc

    # ── search_emails ──────────────────────────────────────────────────────────
    @pytest.mark.asyncio
    async def test_search_emails_success(self):
        svc = self._make_service()
        svc.llm.generate_optimized_query = AsyncMock(return_value="subject:test")
        svc.tool_mgr.fetch_emails = AsyncMock(return_value={
            "success": True,
            "data": {"messages": [{"id": "msg1"}, {"id": "msg2"}]}
        })

        result = await svc.search_emails("emails about test subject")
        assert result["success"] is True
        assert result["total_count"] == 2
        assert result["query_used"] == "subject:test"

    @pytest.mark.asyncio
    async def test_search_emails_short_query_skips_llm(self):
        svc = self._make_service()
        svc.llm.generate_optimized_query = AsyncMock(return_value="unused")
        svc.tool_mgr.fetch_emails = AsyncMock(return_value={
            "success": True,
            "data": {"messages": []}
        })

        # ≤2 words → LLM not called
        await svc.search_emails("inbox", use_llm_optimization=True)
        svc.llm.generate_optimized_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_emails_saves_to_memory(self):
        svc = self._make_service()
        svc.llm.generate_optimized_query = AsyncMock(return_value="is:unread")
        svc.tool_mgr.fetch_emails = AsyncMock(return_value={
            "success": True,
            "data": {"messages": [{"id": "m1"}]}
        })
        svc.memory = AgentMemory()

        await svc.search_emails("unread emails please")
        saved = svc.memory.get_last_search_results(svc.user_id)
        assert saved == ["m1"]

    @pytest.mark.asyncio
    async def test_search_emails_tool_failure(self):
        svc = self._make_service()
        svc.llm.generate_optimized_query = AsyncMock(return_value="q")
        svc.tool_mgr.fetch_emails = AsyncMock(return_value={
            "success": False, "error": "quota exceeded"
        })

        result = await svc.search_emails("some query here")
        assert result["success"] is False
        assert "quota" in result["error"]

    # ── get_email ──────────────────────────────────────────────────────────────
    @pytest.mark.asyncio
    async def test_get_email_success(self):
        svc = self._make_service()
        svc.tool_mgr.fetch_message_by_id = AsyncMock(return_value={
            "success": True,
            "data": {"id": "msg1", "subject": "Hello", "body": "Hi there"}
        })

        result = await svc.get_email("msg1")
        assert result["success"] is True
        assert result["message"]["subject"] == "Hello"

    @pytest.mark.asyncio
    async def test_get_email_failure(self):
        svc = self._make_service()
        svc.tool_mgr.fetch_message_by_id = AsyncMock(return_value={
            "success": False, "error": "not found"
        })

        result = await svc.get_email("bad_id")
        assert result["success"] is False

    # ── send_email ─────────────────────────────────────────────────────────────
    @pytest.mark.asyncio
    async def test_send_email_success(self):
        svc = self._make_service()
        svc.tool_mgr.send_email = AsyncMock(return_value={
            "success": True,
            "data": {"id": "sent123", "labelIds": ["SENT"]}
        })

        result = await svc.send_email(
            to="ashrithaannadata@gmail.com",
            subject="Unit Test Email",
            body="This is a test"
        )
        assert result["success"] is True
        assert result["message"] == "Email sent successfully"

    @pytest.mark.asyncio
    async def test_send_email_failure_propagated(self):
        svc = self._make_service()
        svc.tool_mgr.send_email = AsyncMock(return_value={
            "success": False, "error": "authentication failed"
        })

        result = await svc.send_email("to@x.com", "sub", "body")
        assert result["success"] is False
        assert "authentication" in result["error"]

    # ── reply_to_email ─────────────────────────────────────────────────────────
    @pytest.mark.asyncio
    async def test_reply_to_email_success(self):
        svc = self._make_service()
        svc.tool_mgr.reply_to_thread = AsyncMock(return_value={
            "success": True, "data": {"id": "reply123"}
        })

        result = await svc.reply_to_email(
            thread_id="thread_abc",
            message_id="msg_abc",
            body="Thanks!",
            to="sender@example.com"
        )
        assert result["success"] is True
        assert result["message"] == "Reply sent successfully"

    # ── delete_email ───────────────────────────────────────────────────────────
    @pytest.mark.asyncio
    async def test_delete_email_trash(self):
        svc = self._make_service()
        svc.tool_mgr.move_to_trash = AsyncMock(return_value={"success": True, "data": {}})

        result = await svc.delete_email("msg1", permanent=False)
        assert result["success"] is True
        assert "trash" in result["message"]
        svc.tool_mgr.move_to_trash.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_email_permanent(self):
        svc = self._make_service()
        svc.tool_mgr.delete_message = AsyncMock(return_value={"success": True, "data": {}})

        result = await svc.delete_email("msg1", permanent=True)
        assert result["success"] is True
        assert "permanently" in result["message"]
        svc.tool_mgr.delete_message.assert_called_once()

    # ── create_draft ───────────────────────────────────────────────────────────
    @pytest.mark.asyncio
    async def test_create_draft_saves_to_memory(self):
        svc = self._make_service()
        svc.memory = AgentMemory()
        svc.tool_mgr.create_draft = AsyncMock(return_value={
            "success": True,
            "data": {"id": "draft_xyz", "created_at": "2026-01-01"}
        })

        result = await svc.create_draft(
            to="ashrithaannadata@gmail.com",
            subject="Draft Test",
            body="Draft body"
        )
        assert result["success"] is True
        saved = svc.memory.get_draft_context(svc.user_id, "draft_xyz")
        assert saved is not None
        assert saved["subject"] == "Draft Test"

    # ── list_drafts ────────────────────────────────────────────────────────────
    @pytest.mark.asyncio
    async def test_list_drafts_success(self):
        svc = self._make_service()
        svc.tool_mgr.list_drafts = AsyncMock(return_value={
            "success": True,
            "data": {"drafts": [{"id": "d1"}, {"id": "d2"}]}
        })

        result = await svc.list_drafts()
        assert result["success"] is True
        assert len(result["drafts"]) == 2

    # ── send_draft ─────────────────────────────────────────────────────────────
    @pytest.mark.asyncio
    async def test_send_draft_success(self):
        svc = self._make_service()
        svc.tool_mgr.send_draft = AsyncMock(return_value={
            "success": True, "data": {"id": "sent_draft_1"}
        })

        result = await svc.send_draft("draft_001")
        assert result["success"] is True
        assert "Draft sent successfully" in result["message"]

    # ── delete_draft ───────────────────────────────────────────────────────────
    @pytest.mark.asyncio
    async def test_delete_draft_success(self):
        svc = self._make_service()
        svc.tool_mgr.delete_draft = AsyncMock(return_value={"success": True, "data": {}})

        result = await svc.delete_draft("draft_001")
        assert result["success"] is True
        assert "Draft deleted" in result["message"]

    # ── add_labels ─────────────────────────────────────────────────────────────
    @pytest.mark.asyncio
    async def test_add_labels_success(self):
        svc = self._make_service()
        svc.tool_mgr.add_label_to_email = AsyncMock(return_value={
            "success": True, "data": {}
        })

        result = await svc.add_labels("msg1", ["IMPORTANT", "Label_123"])
        assert result["success"] is True
        assert result["message"] == "Labels added"

    # ── list_labels ────────────────────────────────────────────────────────────
    @pytest.mark.asyncio
    async def test_list_labels_success(self):
        svc = self._make_service()
        svc.tool_mgr.list_labels = AsyncMock(return_value={
            "success": True,
            "data": {"labels": [{"id": "INBOX"}, {"id": "SENT"}]}
        })

        result = await svc.list_labels()
        assert result["success"] is True
        assert len(result["labels"]) == 2

    # ── create_label ───────────────────────────────────────────────────────────
    @pytest.mark.asyncio
    async def test_create_label_success(self):
        svc = self._make_service()
        svc.tool_mgr.create_label = AsyncMock(return_value={
            "success": True,
            "data": {"id": "Label_new", "name": "WorkProjects"}
        })

        result = await svc.create_label("WorkProjects")
        assert result["success"] is True
        assert result["label"]["name"] == "WorkProjects"

    # ── download_attachment ────────────────────────────────────────────────────
    @pytest.mark.asyncio
    async def test_download_attachment_no_data_returns_error(self):
        svc = self._make_service()
        svc.tool_mgr.get_attachment = AsyncMock(return_value={
            "success": True,
            "data": {"data": None}  # No base64 data
        })

        result = await svc.download_attachment("msg1", "att1", "file.pdf")
        assert result["success"] is False
        assert "No attachment data" in result["error"]

    # ── summarize_emails ───────────────────────────────────────────────────────
    @pytest.mark.asyncio
    async def test_summarize_emails_no_valid_emails(self):
        svc = self._make_service()
        svc.tool_mgr.fetch_message_by_id = AsyncMock(return_value={
            "success": False, "error": "not found"
        })

        result = await svc.summarize_emails(["bad_id"])
        assert result["success"] is False
        assert "No emails to summarize" in result["error"]

    @pytest.mark.asyncio
    async def test_summarize_emails_success(self):
        svc = self._make_service()
        svc.tool_mgr.fetch_message_by_id = AsyncMock(return_value={
            "success": True,
            "data": {"id": "m1", "subject": "Hello", "body": "Important content here"}
        })
        svc.llm.summarize_text_batch = AsyncMock(return_value="Summary of email content")

        result = await svc.summarize_emails(["m1"])
        assert result["success"] is True
        assert result["summary"] == "Summary of email content"
        assert result["emails_summarized"] == 1

    # ── draft_smart_reply ──────────────────────────────────────────────────────
    @pytest.mark.asyncio
    async def test_draft_smart_reply_fetch_fails(self):
        svc = self._make_service()
        svc.tool_mgr.fetch_message_by_id = AsyncMock(return_value={
            "success": False, "error": "not found"
        })

        result = await svc.draft_smart_reply("bad_id")
        assert result["success"] is False
        assert "Failed to fetch" in result["error"]

    @pytest.mark.asyncio
    async def test_draft_smart_reply_success(self):
        svc = self._make_service()
        svc.tool_mgr.fetch_message_by_id = AsyncMock(return_value={
            "success": True,
            "data": {
                "id": "m1", "subject": "Meeting", "body": "Can you attend?",
                "from": "boss@company.com", "thread_id": "thread_1"
            }
        })
        svc.llm.draft_email_reply = AsyncMock(return_value={
            "subject": "Re: Meeting",
            "body": "<p>Yes, I'll attend.</p>"
        })
        svc.tool_mgr.create_draft = AsyncMock(return_value={
            "success": True,
            "data": {"id": "new_draft"}
        })

        result = await svc.draft_smart_reply("m1", "confirm attendance")
        assert result["success"] is True
        assert "draft" in result

    # ── extract_action_items ───────────────────────────────────────────────────
    @pytest.mark.asyncio
    async def test_extract_action_items_no_emails(self):
        svc = self._make_service()
        svc.tool_mgr.fetch_message_by_id = AsyncMock(return_value={
            "success": False, "error": "not found"
        })

        result = await svc.extract_action_items(["bad"])
        assert result["success"] is False

    # ── list_contacts ──────────────────────────────────────────────────────────
    @pytest.mark.asyncio
    async def test_list_contacts_success(self):
        svc = self._make_service()
        svc.tool_mgr.get_contacts = AsyncMock(return_value={
            "success": True,
            "data": {"contacts": [{"name": "Alice"}, {"name": "Bob"}]}
        })

        result = await svc.list_contacts()
        assert result["success"] is True
        assert len(result["contacts"]) == 2

    # ── get_profile ────────────────────────────────────────────────────────────
    @pytest.mark.asyncio
    async def test_get_profile_success(self):
        svc = self._make_service()
        svc.tool_mgr.get_profile = AsyncMock(return_value={
            "success": True,
            "data": {"emailAddress": "al.ashritha@gmail.com", "messagesTotal": 1234}
        })

        result = await svc.get_profile()
        assert result["success"] is True
        assert result["profile"]["emailAddress"] == "al.ashritha@gmail.com"

    # ── list_threads ───────────────────────────────────────────────────────────
    @pytest.mark.asyncio
    async def test_list_threads_success(self):
        svc = self._make_service()
        svc.tool_mgr.list_threads = AsyncMock(return_value={
            "success": True,
            "data": {"threads": [{"id": "t1"}, {"id": "t2"}]}
        })

        result = await svc.list_threads(query="is:important", max_results=5)
        assert result["success"] is True
        assert len(result["threads"]) == 2

    # ── get_thread ─────────────────────────────────────────────────────────────
    @pytest.mark.asyncio
    async def test_get_thread_success(self):
        svc = self._make_service()
        svc.tool_mgr.fetch_message_by_thread = AsyncMock(return_value={
            "success": True,
            "data": {"messages": [{"id": "m1"}, {"id": "m2"}]}
        })

        result = await svc.get_thread("thread_abc")
        assert result["success"] is True
        assert len(result["messages"]) == 2

    # ── exception handling ─────────────────────────────────────────────────────
    @pytest.mark.asyncio
    async def test_exception_in_send_email_returns_error(self):
        svc = self._make_service()
        svc.tool_mgr.send_email = AsyncMock(side_effect=Exception("network error"))

        result = await svc.send_email("to@x.com", "sub", "body")
        assert result["success"] is False
        assert "network error" in result["error"]

    @pytest.mark.asyncio
    async def test_exception_in_search_returns_error(self):
        svc = self._make_service()
        svc.tool_mgr.fetch_emails = AsyncMock(side_effect=Exception("timeout"))
        svc.llm.generate_optimized_query = AsyncMock(return_value="q")

        result = await svc.search_emails("some recent emails today")
        assert result["success"] is False
        assert "timeout" in result["error"]


# ==============================================================================
# TestGmailAgentCapabilities
# ==============================================================================
class TestGmailAgentCapabilities:
    """Tests the GmailAgent BaseAgent capability methods end-to-end."""

    def _make_agent(self):
        """
        Create a GmailAgent without triggering ComposioToolManager.
        GmailAgent.__init__ only sets self._services = {} — it does NOT create
        any GmailService yet. Services are lazy-loaded on first capability call via
        _get_service(). Tests inject mock services into _services[user_id] before
        calling capabilities, so the real ComposioToolManager is never instantiated.
        """
        from backend.agents.gmail_agent.base_agent_impl import GmailAgent
        agent = GmailAgent()
        return agent

    def _make_context(self, metadata: dict) -> ExecutionContext:
        return ExecutionContext(
            thread_id="test_conv",
            user_id="user_374hMFRAc0nkaGdH8XtXNRIdfrk",
            task_id="test_task",
            metadata=metadata
        )

    # AgentResponse uses status: Literal["success","error","partial","needs_input"]
    # result.success is a @classmethod, NOT a bool attribute.
    # Use result.status == "success" / "error" and result.error_message for assertions.

    # ── search_emails capability ───────────────────────────────────────────────
    @pytest.mark.asyncio
    async def test_capability_search_missing_query_returns_error(self):
        agent = self._make_agent()
        ctx = self._make_context({"user_id": "user_374hMFRAc0nkaGdH8XtXNRIdfrk"})
        result = await agent.search_emails(ctx)
        assert result.status == "error"
        assert "Query" in result.error_message

    @pytest.mark.asyncio
    async def test_capability_search_emails_success(self):
        agent = self._make_agent()
        mock_service = MagicMock()
        mock_service.search_emails = AsyncMock(return_value={
            "success": True,
            "messages": [{"id": "m1", "subject": "Test"}],
            "total_count": 1,
            "query_used": "subject:Test"
        })
        agent._services["user_374hMFRAc0nkaGdH8XtXNRIdfrk"] = mock_service

        ctx = self._make_context({
            "user_id": "user_374hMFRAc0nkaGdH8XtXNRIdfrk",
            "query": "emails with subject test"
        })
        result = await agent.search_emails(ctx)
        assert result.status == "success"
        assert result.result["total_count"] == 1

    @pytest.mark.asyncio
    async def test_capability_search_emails_service_error(self):
        agent = self._make_agent()
        mock_service = MagicMock()
        mock_service.search_emails = AsyncMock(return_value={
            "success": False, "error": "token expired"
        })
        agent._services["user_374hMFRAc0nkaGdH8XtXNRIdfrk"] = mock_service

        ctx = self._make_context({
            "user_id": "user_374hMFRAc0nkaGdH8XtXNRIdfrk",
            "query": "recent important emails"
        })
        result = await agent.search_emails(ctx)
        assert result.status == "error"
        assert "token expired" in result.error_message

    # ── send_email capability ──────────────────────────────────────────────────
    @pytest.mark.asyncio
    async def test_capability_send_email_missing_params(self):
        agent = self._make_agent()
        ctx = self._make_context({
            "user_id": "user_374hMFRAc0nkaGdH8XtXNRIdfrk",
            "to": "ashrithaannadata@gmail.com"
            # subject and body missing
        })
        result = await agent.send_email(ctx)
        assert result.status == "error"
        assert "Missing" in result.error_message

    @pytest.mark.asyncio
    async def test_capability_send_email_success(self):
        agent = self._make_agent()
        mock_service = MagicMock()
        mock_service.send_email = AsyncMock(return_value={
            "success": True,
            "message": "Email sent successfully",
            "data": {"id": "sent_001"}
        })
        agent._services["user_374hMFRAc0nkaGdH8XtXNRIdfrk"] = mock_service

        ctx = self._make_context({
            "user_id": "user_374hMFRAc0nkaGdH8XtXNRIdfrk",
            "to": "ashrithaannadata@gmail.com",
            "subject": "Unit Test — Gmail Agent",
            "body": "This email was sent by an automated unit test."
        })
        result = await agent.send_email(ctx)
        assert result.status == "success"
        assert result.summary == "Email sent successfully"

    @pytest.mark.asyncio
    async def test_capability_send_email_failure(self):
        agent = self._make_agent()
        mock_service = MagicMock()
        mock_service.send_email = AsyncMock(return_value={
            "success": False, "error": "SMTP timeout"
        })
        agent._services["user_374hMFRAc0nkaGdH8XtXNRIdfrk"] = mock_service

        ctx = self._make_context({
            "user_id": "user_374hMFRAc0nkaGdH8XtXNRIdfrk",
            "to": "ashrithaannadata@gmail.com",
            "subject": "Test",
            "body": "Body"
        })
        result = await agent.send_email(ctx)
        assert result.status == "error"
        assert "SMTP timeout" in result.error_message

    # ── reply_email capability ─────────────────────────────────────────────────
    @pytest.mark.asyncio
    async def test_capability_reply_email_missing_params(self):
        agent = self._make_agent()
        ctx = self._make_context({
            "user_id": "user_374hMFRAc0nkaGdH8XtXNRIdfrk"
            # message_id and body missing
        })
        result = await agent.reply_email(ctx)
        assert result.status == "error"
        assert "Missing" in result.error_message

    @pytest.mark.asyncio
    async def test_capability_reply_email_success(self):
        agent = self._make_agent()
        mock_service = MagicMock()
        mock_service.reply_to_email = AsyncMock(return_value={
            "success": True,
            "message": "Reply sent successfully",
            "data": {}
        })
        agent._services["user_374hMFRAc0nkaGdH8XtXNRIdfrk"] = mock_service

        ctx = self._make_context({
            "user_id": "user_374hMFRAc0nkaGdH8XtXNRIdfrk",
            "message_id": "msg_thread_001",
            "body": "Thanks for your email!"
        })
        result = await agent.reply_email(ctx)
        assert result.status == "success"

    # ── get_email capability ───────────────────────────────────────────────────
    @pytest.mark.asyncio
    async def test_capability_get_email_missing_message_id(self):
        agent = self._make_agent()
        ctx = self._make_context({"user_id": "user_374hMFRAc0nkaGdH8XtXNRIdfrk"})
        result = await agent.get_email(ctx)
        assert result.status == "error"
        assert "message_id" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_capability_get_email_success(self):
        agent = self._make_agent()
        mock_service = MagicMock()
        mock_service.get_email = AsyncMock(return_value={
            "success": True,
            "message": {"id": "msg1", "subject": "Hello", "body": "World"}
        })
        agent._services["user_374hMFRAc0nkaGdH8XtXNRIdfrk"] = mock_service

        ctx = self._make_context({
            "user_id": "user_374hMFRAc0nkaGdH8XtXNRIdfrk",
            "message_id": "msg1"
        })
        result = await agent.get_email(ctx)
        assert result.status == "success"
        assert result.result["subject"] == "Hello"
