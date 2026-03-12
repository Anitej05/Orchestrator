"""
Unit tests for backend/agents/base/agent.py and capability.py.

Covers the BaseAgent architecture fixes that prevent swallowed AgentResponse
states and redundant ReAct loops.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from backend.agents.base.agent import BaseAgent, AgentConfig
from backend.agents.base.capability import capability
from backend.agents.base.types import AgentRequest, AgentResponse


class _FakeInference:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def generate_structured(self, messages, schema, temperature=0.0):
        self.calls.append(
            {
                "schema": schema.__name__,
                "messages": messages,
                "temperature": temperature,
            }
        )
        payload = self._responses.pop(0)
        return schema(**payload)


class _FakeTelemetry:
    def log_agent_call(self, *args, **kwargs):
        return None

    def log_error(self, *args, **kwargs):
        return None


class _FakeCredentials:
    def get(self, *args, **kwargs):
        return None

    def get_all(self, *args, **kwargs):
        return {}


class _DummyServices:
    def __init__(self, inference):
        self.inference = inference
        self.telemetry = _FakeTelemetry()
        self.credentials = _FakeCredentials()

    def initialize_essential(self):
        return None


class _DummyAgent(BaseAgent):
    def __init__(self, services):
        self.fetch_calls = 0
        self.load_calls = 0
        self.summarize_calls = 0
        self.ask_calls = 0
        super().__init__(
            agent_id="dummy_agent",
            agent_name="Dummy Agent",
            services=services,
            config=AgentConfig(),
        )

    async def _initialize_resources(self):
        return None

    @capability(
        name="fetch_data",
        description="Fetch a final answer payload in one step.",
    )
    async def fetch_data(self, params, context):
        self.fetch_calls += 1
        return {
            "success": True,
            "data": {"items": [{"id": 1, "value": "ok"}]},
            "message": "Fetched data",
        }

    @capability(
        name="load_file",
        description="Load a file into working memory.",
    )
    async def load_file(self, params, context):
        self.load_calls += 1
        return {
            "success": True,
            "data": {"file_id": "file_123"},
            "message": "Loaded file",
        }

    @capability(
        name="summarize_data",
        description="Summarize loaded data into a final answer.",
    )
    async def summarize_data(self, params, context):
        self.summarize_calls += 1
        return {
            "success": True,
            "data": {"summary": "All done"},
            "message": "Summary ready",
            "final": True,
        }

    @capability(
        name="ask_user",
        description="Request missing information from the user.",
    )
    async def ask_user(self, params, context):
        self.ask_calls += 1
        return AgentResponse.needs_input(
            question="Which account should I use?",
            question_type="text",
        )


def _request(prompt: str, action=None) -> AgentRequest:
    return AgentRequest(
        prompt=prompt,
        action=action,
        payload={},
        task_id="task_1",
        thread_id="thread_1",
        user_id="user_1",
    )


@pytest.mark.asyncio
async def test_direct_action_bubbles_needs_input():
    agent = _DummyAgent(services=_DummyServices(_FakeInference([])))

    resp = await agent.execute(_request("Ask for clarification", action="ask_user"))

    assert resp.status == "needs_input"
    assert resp.question == "Which account should I use?"
    assert agent.ask_calls == 1


@pytest.mark.asyncio
async def test_react_auto_finishes_after_single_terminal_step():
    inference = _FakeInference(
        [
            {
                "intent": "Fetch unread items",
                "entities": {},
                "implicit_needs": [],
                "complexity": "simple",
                "confidence": 0.99,
            },
            {
                "reasoning": "A single fetch is enough.",
                "capability_name": "fetch_data",
                "description": "Fetch the data now.",
                "parameters": {},
                "expected_outcome": "Return the requested data.",
            },
        ]
    )
    agent = _DummyAgent(services=_DummyServices(inference))

    resp = await agent.execute(_request("Fetch the data and return it"))

    assert resp.status == "success"
    assert agent.fetch_calls == 1
    assert len(inference.calls) == 2
    assert resp.result == {"items": [{"id": 1, "value": "ok"}]}


@pytest.mark.asyncio
async def test_react_does_not_finish_after_non_terminal_load_step():
    inference = _FakeInference(
        [
            {
                "intent": "Load then summarize",
                "entities": {"file": "report.csv"},
                "implicit_needs": [],
                "complexity": "medium",
                "confidence": 0.95,
            },
            {
                "reasoning": "Need to load the file first.",
                "capability_name": "load_file",
                "description": "Load the file.",
                "parameters": {"file_path": "report.csv"},
                "expected_outcome": "File is loaded.",
            },
            {
                "reasoning": "Now summarize the loaded file.",
                "capability_name": "summarize_data",
                "description": "Summarize the loaded data.",
                "parameters": {},
                "expected_outcome": "Return the summary.",
            },
        ]
    )
    agent = _DummyAgent(services=_DummyServices(inference))

    resp = await agent.execute(_request("Load report.csv and summarize it"))

    assert resp.status == "success"
    assert agent.load_calls == 1
    assert agent.summarize_calls == 1
    assert len(inference.calls) == 3
