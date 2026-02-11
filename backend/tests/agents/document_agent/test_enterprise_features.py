"""
Enterprise-grade feature tests for Document Agent.
Focused on safety gating, NEEDS_INPUT, phase tracing, grounding, and state persistence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

import pytest

from backend.agents.document_agent.agent import DocumentAgent
from backend.agents.document_agent.schemas import EditDocumentRequest, AnalyzeDocumentRequest
from backend.agents.document_agent.state import DialogueStateManager, DialogueRecord


@pytest.fixture()
def temp_docx(tmp_path: Path):
    docx = pytest.importorskip("docx")
    file_path = tmp_path / "sample.docx"
    document = docx.Document()
    document.add_paragraph("Hello World")
    document.save(str(file_path))
    return file_path


def test_intent_classification_risk_scoring():
    agent = DocumentAgent()
    low = agent._classify_edit_intent("Add a heading at the end")
    high = agent._classify_edit_intent("Delete all content in this file")
    assert 0.0 <= low["risk_score"] <= 1.0
    assert 0.0 <= high["risk_score"] <= 1.0
    assert high["risk_score"] >= low["risk_score"]
    assert high["intent"] in {"destructive", "overwrite"}


def test_plan_validation_enforces_limits():
    agent = DocumentAgent()
    plan = {
        "actions": [{"type": "add_paragraph", "text": "x"} for _ in range(26)]
    }
    result = agent._validate_edit_plan(plan)
    assert result["valid"] is False
    assert any("26" in issue or ">" in issue for issue in result["issues"])


def test_needs_input_pause_for_risky_edit(temp_docx: Path, monkeypatch: pytest.MonkeyPatch):
    agent = DocumentAgent()

    def fake_plan(*_args, **_kwargs):
        return {
            "success": True,
            "actions": [
                {"type": "delete_content", "target": "all"}
            ],
        }

    monkeypatch.setattr(agent.llm_client, "interpret_edit_instruction", fake_plan)

    request = EditDocumentRequest(
        file_path=str(temp_docx),
        instruction="Delete all content",
        thread_id="test-thread"
    )
    result = agent.edit_document(request)
    assert result.get("status") == "needs_input"
    assert result.get("question")
    assert result.get("pending_plan") is not None


def test_edit_verification_noop(monkeypatch: pytest.MonkeyPatch, temp_docx: Path):
    agent = DocumentAgent()

    def fake_plan(*_args, **_kwargs):
        return {"success": True, "actions": []}

    monkeypatch.setattr(agent.llm_client, "interpret_edit_instruction", fake_plan)

    # Force identical hashes to simulate no-op
    hashes = ["abc", "abc"]

    def fake_hash(*_args, **_kwargs):
        return hashes.pop(0)

    monkeypatch.setattr(agent, "_hash_file_md5", fake_hash)

    request = EditDocumentRequest(
        file_path=str(temp_docx),
        instruction="Do nothing",
        thread_id="test-thread"
    )
    result = agent.edit_document(request)
    assert result.get("edit_summary", {}).get("reason") == "no_change_detected"
    assert result.get("status") == "complete"


def test_phase_trace_in_edit(temp_docx: Path, monkeypatch: pytest.MonkeyPatch):
    agent = DocumentAgent()

    def fake_plan(*_args, **_kwargs):
        return {"success": True, "actions": []}

    monkeypatch.setattr(agent.llm_client, "interpret_edit_instruction", fake_plan)

    request = EditDocumentRequest(
        file_path=str(temp_docx),
        instruction="No changes",
        thread_id="test-thread"
    )
    result = agent.edit_document(request)
    phases = result.get("phase_trace")
    assert isinstance(phases, list)
    assert "understand" in phases
    assert "report" in phases


def test_grounding_confidence_when_vector_store_present(tmp_path: Path):
    # Skip if no vector store available
    storage = Path(__file__).parent.parent.parent.parent / "storage" / "vector_store"
    stores = list(storage.glob("*.faiss")) if storage.exists() else []
    if not stores:
        pytest.skip("No vector store available for grounding test")

    agent = DocumentAgent()
    test_file = None
    # pick any available document file from test data
    test_data_dir = Path(__file__).parent / "test_data"
    for ext in ("*.pdf", "*.docx", "*.txt"):
        files = list(test_data_dir.glob(ext))
        if files:
            test_file = files[0]
            break

    if not test_file:
        pytest.skip("No test document available for grounding test")

    request = AnalyzeDocumentRequest(
        file_path=str(test_file),
        vector_store_path=str(stores[0]),
        query="What is the title?",
        thread_id="test-grounding"
    )
    result = agent.analyze_document(request)
    assert result.get("grounding") is not None
    assert result.get("confidence") is not None


def test_dialogue_state_persistence(tmp_path: Path):
    db_path = tmp_path / "dialogue_state.db"
    manager = DialogueStateManager(db_path=str(db_path))

    record = manager.get_or_create("task-1", "document_agent")
    assert record.task_id == "task-1"

    manager.set_question("task-1", {"question": "Approve?", "status": "needs_input"})
    loaded = manager.get("task-1")
    assert loaded.status == "paused"
    assert loaded.current_question["question"] == "Approve?"

    manager.update_status("task-1", "completed")
    loaded = manager.get("task-1")
    assert loaded.status == "completed"
